import rclpy
import os
import csv
import time
import torch
import random
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pointcloud2 as pc2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from rclpy.node import Node
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from ros_gz_interfaces.srv import ControlWorld
from functools import partial
from collections import deque
from squaternion import Quaternion

global x_endPos
global y_endPos
global x_tugPos
global y_tugPos
global ang_tug #Note this value is in radians
global dist_list
global distance_to_end
global laser_read
global pose_called
global laser_called
global odom_called

COLLISION_DISTANCE = 0.8
DEAD_END_DISTANCE = 2.5 #2.5 for v1-2
END_REACH_DISTANCE = 2.0#1.5 for v1-2
STEP_TIME = 0.25 #this is in seconds use 0.25 for v1-2 / 0.1 for v1-1
LR_CRITIC = 0.001 #change to 0.001
LR_ACTOR = 0.001
GAMMA = 0.9999
DECAY_WEIGHT = 0.01
TAU = 0.005 # change to 0.005
BUFFER_SIZE = 1000000
MINI_BATCH = 510 #change back to 510
MAX_ACTION = 1
EXPL_DECAY_STEPS = 80000
EXPL_MIN = 0.1
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Note: Dimension of the state is now 15 + 4 for the robot due to the laser only having 15 readings

class gazebo_Interface(Node):

    def __init__(self):
        super().__init__("ml_node")

        #Initialize the communications to gazebo
        self.get_logger().info("Ml Node Started")
        self.end_point_pos =self.create_subscription(Pose, "/model/end_point/pose",self.end_pose_callback, 10)#add msg type, topic name, (info) callback method, 10 
        self.get_logger().info("End_point Subscriber has Started")
        self.robot_pos = self.create_subscription(Pose,"/model/tugbot/pose",self.robot_pose_callback, 10)
        self.get_logger().info("Robot_Pose Subscriber has Started")
        self.robot_pos = self.create_subscription(PointCloud2,"/world/world_demo/model/tugbot/link/scan_front/sensor/scan_front/scan/points",self.lidar_scan_callback, 20)
        self.get_logger().info("Robot_Pose Subscriber has Started")
        self.robot_cmd = self.create_publisher(Twist,"/model/tugbot/cmd_vel", 10)#add the msg_type, topic name, 10
        self.get_logger().info("PubNode has Started")
        self.robot_odom = self.create_subscription(Odometry, "/moel/tugbot/odometry", self.robot_odom_callback, 10)
        self.get_logger().info("Odom Node has been started")

        self.state_size = 19
        self.action_size = 2
        self.laser_read = [0]
        self.x_tugPos = 0
        self.y_tugPos = 0
        self.ang_tug = 0
        self.x_endPos = 5
        self.y_endPos = 5
        self.pose_called = False
        self.odom_called = False
        self.laser_called = False
        self.publish_called = False
        self.distance_to_end = 50
        self.prev_distance = 50
        self.expl_noise = 0.327#Normally 1
        self.dist_list = []
        self.count_rand_actions = 0
        self.random_action = []
        self.loss_a = []
        self.loss_c = []

        #Initialize the Actor Networks
        self.local_actor = Actor(self.state_size).to(device)
        self.target_actor = Actor(self.state_size).to(device)
        self.target_actor.load_state_dict(self.local_actor.state_dict())
        self.opt_actor = optim.Adam(self.local_actor.parameters())
        

        #Initialize the Critic Networks
        self.local_critic = Critic(self.state_size).to(device)
        self.target_critic = Critic(self.state_size).to(device)
        self.target_critic.load_state_dict(self.local_critic.state_dict())
        self.opt_critic = optim.Adam(self.local_critic.parameters(), lr=LR_CRITIC)
        
        #Initialize the Replay Memories
        self.memory = Replay_Buffer(BUFFER_SIZE)
           
    def end_pose_callback(self, msg: Pose):
        self.x_endPos = float(msg.position.x)
        self.y_endPos = float(msg.position.y)
    
    def robot_pose_callback(self, msg: Pose):
       
       self.x_tugPos = float(msg.position.x)
       self.y_tugPos = float(msg.position.y)

       quaternion = Quaternion(
        msg.orientation.w,
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        )
       euler = quaternion.to_euler(degrees=False)
       self.ang_tug = round(euler[2], 4)

       self.heading_Calc()
       self.pose_called = True
       
       x_dist = self.x_tugPos - self.x_endPos
       y_dist = self.y_tugPos - self.y_endPos
       self.distance_to_end = np.sqrt(x_dist*x_dist + y_dist*y_dist)

    def lidar_scan_callback(self, msg: PointCloud2):
        self.laser_read = list(pc2.read_points(msg, skip_nans=False))
        self.dist_list = []
        for i in range(0,len(self.laser_read),1):
            pt = self.laser_read[i]
            if i%45 ==0:
                dist = np.sqrt(pt[0]*pt[0] + pt[1]*pt[1])
                if dist == np.Inf:
                    self.dist_list.append(10)
                else:
                    self.dist_list.append(dist)
        self.laser_called = True
    
    def robot_odom_callback(self, msg: Odometry):
        self.odom_called = True
    
    def cmd_pub_callback(self):
        self.publish_called = True
 
    def call_reset(self, node, i, steps):
       
        actionreset = [0,0]
        cmd = Twist() 
        cmd.linear.x = float(actionreset[0])
        cmd.angular.z = float(actionreset[1])
        self.robot_cmd.publish(cmd)
        self.pose_called = False
        self.odom_called = False
        self.laser_called = False  

        if i <= (steps//4) :
            os.system("ign service -s /world/world_demo/set_pose --reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean --timeout 300 --req 'name:\"tugbot\", position: {x: 7, y: 4, z: 0} orientation: {x: 0.0, y: 0.0, z: 1, w: 0.0000463}'")

        elif i > (steps//4) and i <= (steps//2):
            os.system("ign service -s /world/world_demo/set_pose --reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean --timeout 300 --req 'name:\"tugbot\", position: {x: 8, y: 19, z: 0} orientation: {x: 0.0, y: 0.0, z: 0.0, w: 0.5}'")

        elif i > (steps//2) and i <=(steps*3//4):
            os.system("ign service -s /world/world_demo/set_pose --reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean --timeout 300 --req 'name:\"tugbot\", position: {x: -8, y: 4, z: 0} orientation: {x: 0.0, y: 0.0, z: 0.0, w: 0.5}'")
        
        else: 
            os.system("ign service -s /world/world_demo/set_pose --reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean --timeout 300 --req 'name:\"tugbot\", position: {x: 7, y: 4, z: 0} orientation: {x: 0.0, y: 0.0, z: 0.0, w: 0.5}'")
        time.sleep(0.5)
        
        while self.pose_called == False and self.odom_called == False and self.laser_called == False:
            rclpy.spin_once(node)
        
        rclpy.spin_once(node)

        laser_state = self.dist_list
        dist_to_end = self.distance_to_end
        action = [0,0]
        
        robot_state = [dist_to_end, self.heading_Calc(), action[0], action[1]]
        state = laser_state + robot_state
        state = np.array(state)
        return state
    
    def call_pause(self, ps_pl: bool):
        client = self.create_client(ControlWorld,"/world/world_demo/control")
        while not client.wait_for_service(1.0):
            self.get_logger().warn("Waiting for service...")

        request = ControlWorld.Request()
        request.world_control.pause = ps_pl

        future = client.call_async(request)
        future.add_done_callback(partial(self.callback_set_pause))
    
    def callback_set_pause(self, future):
        try:
            response = future.result()
        except:
            self.get_logger().error("Service call failed")

    def dead_end_check(self, dist: []):
        end = max(dist)
        if end <= DEAD_END_DISTANCE:
            return True
        return False
    
    def collision_detect(self, dist: []):
        coll = min(dist[2:-2])
        if coll <= COLLISION_DISTANCE:
            return True
        return False
    
    def end_detect(self):
        if self.distance_to_end <= END_REACH_DISTANCE:
            return True
        return False

    def get_reward(self, endPoint: bool, collision: bool, deadEnd: bool):
        temp_reward = 0
        if endPoint == True:
            temp_reward = 1000
        elif collision:
            collision = False
            temp_reward = -1
        #elif deadEnd:
            #deadEnd = False
            #return -100
        elif self.distance_to_end < self.prev_distance:
            temp_reward = 1

        self.prev_distance = self.distance_to_end
        return temp_reward

    def heading_Calc(self):

        skew_x = self.x_endPos - self.x_tugPos
        skew_y = self.y_endPos - self.y_tugPos
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if self.ang_tug < 0:
            theta1 = 2*np.pi - (beta - self.ang_tug)
            theta2 = beta - self.ang_tug
            theta = min(theta1,theta2)
        elif beta < self.ang_tug:
            theta = self.ang_tug - beta
        else:
            theta = beta - self.ang_tug

        return theta

    def step(self, action, node):

        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])
        self.robot_cmd.publish(cmd)

        self.pose_called = False
        self.odom_called = False
        self.laser_called = False
        self.publish_called = False

        while self.pose_called == False and self.odom_called == False and self.laser_called == False:
            rclpy.spin_once(node)

        self.call_pause(False)
        time.sleep(STEP_TIME)
        cmd.linear.x = float(0)
        cmd.angular.z = float(0)
        self.robot_cmd.publish(cmd)
        self.call_pause(True)
        time.sleep(0.5)

        self.pose_called = False
        self.odom_called = False
        self.laser_called = False

        while self.pose_called == False and self.odom_called == False and self.laser_called == False:
            rclpy.spin_once(node)
       
        laser_state = self.dist_list
        dist_to_end = self.distance_to_end

        alpha = self.heading_Calc()
        complete = self.end_detect()

        state_robot = [dist_to_end, alpha, action[0], action[1]]
        state = laser_state + state_robot
        reward = self.get_reward(self.end_detect(),self.collision_detect(laser_state),self.dead_end_check(laser_state))

        return state , reward, complete

    def mem_step(self, state, action, reward, next_state, complete):
        self.memory.add_experience(state, action, reward, next_state, int(complete))#convert complete to int Test8
        if self.memory.get_size() > MINI_BATCH:
            experience = self.memory.sample_ranBatch(40)
            self.learn(experience)
    
    def act(self, state):
        state = torch.Tensor(state).to(device)
        action = self.local_actor.forward_pass(state)
        return action
    
    def learn(self, experiences, gamma=GAMMA):

        states, actions, rewards, next_states, completes = experiences

        state = torch.Tensor(states).to(device)
        action = torch.Tensor(actions).to(device)
        reward = torch.Tensor(rewards).to(device)
        next_state = torch.Tensor(next_states).to(device)
        complete = torch.Tensor(completes).to(device)

        actions_next = self.target_actor.forward(next_state)

        noise = torch.Tensor(actions).data.normal_(0, 0.2).to(device)
        noise = noise.clamp(-0.5,0.5)
        actions_next = torch.Tensor(actions_next)
        actions_next = (actions_next + noise).clamp(-1,1)
        actions_next = np.array(actions_next)

        next_targets_Q = self.target_critic.forward_pass(next_state, torch.transpose(torch.Tensor(np.array(actions_next))[0:2,0:511,0],0,1))
        targets_Q = reward + (gamma*next_targets_Q*(1-complete))

        expected_Q = self.local_critic.forward_pass(state, action)
        loss_critic = F.mse_loss(expected_Q, targets_Q)
        self.loss_c.append(loss_critic)

        self.opt_critic.zero_grad()
        loss_critic.backward()
        self.opt_critic.step()

        pred_actions = self.local_actor.forward(state)
        actor_grad = self.local_critic.forward_pass(state,torch.transpose(torch.Tensor(np.array(pred_actions))[0:2,0:511,0],0,1))
        loss_actor = -actor_grad.mean()
        self.loss_a.append(loss_actor)

        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        #Update the target networks
        self.soft_update(self.local_actor, self.target_actor, TAU)
        self.soft_update(self.local_critic, self.target_critic, TAU)
    
    def soft_update(self, model_local, model_target , tau):
        for target_param, local_param in zip(model_target.parameters(), model_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
                
    def ran_action(self, node, state):
        
        action = [0,0]

        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[3:-5]) < 0.8
            and self.count_rand_actions < 1
        ):
            self.count_rand_actions = np.random.randint(8, 15)
            self.random_action = np.random.uniform(-1, 1, 2)
            action = self.random_action

        if self.count_rand_actions > 0:
            self.count_rand_actions -= 1
            action = self.random_action
            action[0] = -1
        
        return action

class Replay_Buffer(object):
    def __init__(self, size):

        self.size_buff = size
        self.counter = 0
        self.buffer = deque()
        random.seed(0)
    
    def add_experience(self, state, action, reward, next_state, complete):
        exp = (state, action, reward, next_state, complete)
        if self.counter < self.size_buff:
            self.buffer.append(exp)
            self.counter += 1
        else:
            self.buffer.popleft()
            self.buffer.append(exp)
    
    def get_size(self):
        return self.counter
    
    def sample_ranBatch(self, size_batch):
        batch = []
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = [] 
        complete = []

        if self.counter < size_batch:
            batch = random.sample(self.buffer,self.counter)
        else:
            batch = random.sample(self.buffer,size_batch)

        state_batch = np.array([_[0] for _ in batch])
        action_batch = np.array([_[1] for _ in batch])
        reward_batch = np.array([_[2] for _ in batch]).reshape(-1,1)
        next_state_batch = np.array([_[3] for _ in batch])
        complete = np.array([_[4] for _ in batch]).reshape(-1,1)

        return state_batch, action_batch, reward_batch, next_state_batch, complete
        
    def clear_buffer(self):
        self.buffer.clear()
        self.counter = 0

class Actor(nn.Module):
    def __init__(self, dim_state):

        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(dim_state,512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer_2 = nn.Linear(512,512)
        self.layer_3 = nn.Linear(512,512)
        self.layer_4_Lin = nn.Linear(512,1)
        self.layer_4_Ang = nn.Linear(512,1)
    
    def forward_pass(self, state):

        state = F.relu(self.layer_1(state))
        state = F.relu(self.layer_2(state))
        state = F.relu(self.layer_3(state))
        lin_vel = F.sigmoid(self.layer_4_Lin(state)).cpu().detach().numpy()
        lin_vel = float(lin_vel)
        ang_vel = F.tanh(self.layer_4_Ang(state)).cpu().detach().numpy()
        ang_vel = float(ang_vel)
        return [lin_vel, ang_vel]
    
    def forward(self, state):

        state = F.relu(self.layer_1(state))
        state = F.relu(self.layer_2(state))
        state = F.relu(self.layer_3(state))
        lin_vel = F.sigmoid(self.layer_4_Lin(state)).cpu().detach().numpy()
        ang_vel = F.tanh(self.layer_4_Ang(state)).cpu().detach().numpy()
        return [lin_vel, ang_vel]
    
    def reset_parameters(self):

        #Reset the weights of the layer using a uniform distribution based on the initial weight data of the layer

        f1 = 1./np.sqrt(self.layer_1.weight.data.size()[0])
        self.layer_1.weight.data.uniform_(-f1,f1)
        self.layer_1.bias.data.uniform_(-f1,f1)

        f2 = 1./np.sqrt(self.layer_2.weight.data.size()[0])
        self.layer_2.weight.data.uniform_(-f2,f2)
        self.layer_2.bias.data.uniform_(-f2,f2)

        f3 = 1./np.sqrt(self.layer_3.weight.data.size()[0])
        self.layer_3.weight.data.uniform_(-f3,f3)
        self.layer_3.bias.data.uniform_(-f3,f3)

        f4_lin = 1./np.sqrt(self.layer_4_Lin.weight.data.size()[0])
        self.layer_4_Lin.weight.data.uniform_(-f4_lin,f4_lin)
        self.layer_4_Lin.bias.data.uniform_(-f4_lin,f4_lin)

        f4_ang = 1./np.sqrt(self.layer_4_Ang.weight.data.size()[0])
        self.layer_4_Ang.weight.data.uniform_(-f4_ang,f4_ang)
        self.layer_4_Ang.bias.data.uniform_(-f4_ang,f4_ang)

class Critic(nn.Module):
    def __init__(self, dim_state):

        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(dim_state,512)
        self.layer_2 = nn.Linear(514,512)
        self.layer_3 = nn.Linear(512,512)
        self.layer_4 = nn.Linear(512,1)
    
    def forward_pass(self, state, action):

        state = F.relu(self.layer_1(state))
        state = torch.cat((state,action), dim =1)
        state = F.relu(self.layer_2(state))
        state = F.relu(self.layer_3(state))
        Q_value = self.layer_4(state)
        return Q_value
    
    def reset_parameters(self):
        
        f1 = 1./np.sqrt(self.layer_1.weight.data.size()[0])
        self.layer_1.weight.data.uniform_(-f1,f1)
        self.layer_1.bias.data.uniform_(-f1,f1)

        f2 = 1./np.sqrt(self.layer_2.weight.data.size()[0])
        self.layer_2.weight.data.uniform_(-f2,f2)
        self.layer_2.bias.data.uniform_(-f2,f2)

        f3 = 1./np.sqrt(self.layer_3.weight.data.size()[0])
        self.layer_3.weight.data.uniform_(-f3,f3)
        self.layer_3.bias.data.uniform_(-f3,f3)

        f4 = 1./np.sqrt(self.layer_4.weight.data.size()[0])
        self.layer_4.weight.data.uniform_(-f4,f4)
        self.layer_4.bias.data.uniform_(-f4,f4)

def train(episodes, step, pretrained=False, New_run=False):

    robot = gazebo_Interface()
    torch.manual_seed(0)  
    np.random.seed(0) 

    if pretrained:
        robot.local_actor.load_state_dict(torch.load('src/tugbot_ml_controller/models/weights/checkpoint_actor.pth', map_location="cpu"))
        robot.target_actor.load_state_dict(torch.load('src/tugbot_ml_controller/models/weights/checkpoint_actor_t.pth', map_location="cpu"))
        robot.local_critic.load_state_dict(torch.load('src/tugbot_ml_controller/models/weights/checkpoint_critic.pth', map_location="cpu"))
        robot.target_critic.load_state_dict(torch.load('src/tugbot_ml_controller/models/weights/checkpoint_critic_t.pth', map_location="cpu"))

        with open('src/tugbot_ml_controller/CSV_Files/Buffer.csv','r') as File:
            csvreader = csv.reader(File)
            header = next(csvreader)
            for row in csvreader:
                states = data_manip(row[0], 0).astype('float16')
                actions = data_manip(row[1],1).astype('float16')
                reward_exp = int(row[2])
                Next_state = data_manip(row[3],3).astype('float16')
                completes = bool(row[4])

                if len(actions) != 2:
                    actions = data_manip(row[1],5).astype('float16')

                if len(Next_state) != 19:
                    Next_state = data_manip(row[3],5).astype('float16')
                
                robot.memory.add_experience(states,actions,reward_exp,Next_state,int(completes))
        print("Load Succesful")
    
    if New_run:
        File = open('src/tugbot_ml_controller/CSV_Files/Buffer.csv','w+', newline='')
        File.truncate()
        File.close()
    
    list_rewards = []
    list_time = []
    prev_score = 0
    

    for i in range(1,episodes+1):

        robot.pose_called = False
        robot.odom_called = False
        robot.laser_called = False
        robot.prev_distance = 50
        coll_count = 0

        while robot.pose_called == False and robot.odom_called == False and robot.laser_called == False:
            rclpy.spin_once(robot)
        
        state = robot.call_reset(robot, 1, step)#change 1 to i if changing the starting point throughout the training
        time.sleep(2)
        score = 0
        complete = False

        robot.pose_called = False
        robot.odom_called = False
        robot.laser_called = False

        while robot.pose_called == False & robot.odom_called == False & robot.laser_called == False:
            rclpy.spin_once(robot)
        time.sleep(0.5)

        for t in range(step):


            action = robot.act(state)
            complete = False
            

            if robot.expl_noise > EXPL_MIN:
                robot.expl_noise = robot.expl_noise - ((1 - EXPL_MIN)/EXPL_DECAY_STEPS)

            action = (action + np.random.normal(0,robot.expl_noise,2)).clip(-MAX_ACTION,MAX_ACTION)
            
            if robot.collision_detect(robot.dist_list) == True:
                action = robot.ran_action(robot, state)

            action = [(action[0]+1)/2,action[1]] 
            next_state, reward, complete = robot.step(action, robot)
            robot.mem_step(state, action, reward, next_state, complete)
            state = np.array(next_state)
            score += reward    

            if complete:
                if i%2 == 0:
                    print('Reward: {} | Episode: {}/{}'.format(np.array(list_rewards)[-10:].mean(), i, episodes))
                    print(f"Timesteps: {t}. Time (sec): {format(np.array(list_time)[-10:].mean()/50, '.3f')}")
                    complete = False
                complete = False
                break

            if t % 20 == 0:
                robot.target_actor.load_state_dict(robot.local_actor.state_dict())
                robot.target_critic.load_state_dict(robot.local_critic.state_dict())

        

        list_rewards.append(score)
        list_time.append(t)

        if(i%2 == 0):
            print(f"\nMEAN REWARD: {np.array(list_rewards)[-100:].mean()}\n")
            torch.save(robot.local_actor.state_dict(), 'src/tugbot_ml_controller/checkpoints/checkpoint_actor_'+str("%03d" % (i//2))+'.pth')
            torch.save(robot.local_critic.state_dict(), 'src/tugbot_ml_controller/checkpoints/checkpoint_critic_'+str("%03d" % (i//2))+'.pth')
            torch.save(robot.target_actor.state_dict(), 'src/tugbot_ml_controller/checkpoints/checkpoint_actor_t_'+str("%03d" % (i//2))+'.pth')
            torch.save(robot.target_critic.state_dict(), 'src/tugbot_ml_controller/checkpoints/checkpoint_critic_t_'+str("%03d" % (i//2))+'.pth')

        if score > prev_score:
            torch.save(robot.local_actor.state_dict(), 'src/tugbot_ml_controller/models/weights/best_actor.pth')
            torch.save(robot.local_critic.state_dict(), 'src/tugbot_ml_controller/models/weights/best_critic.pth')
            torch.save(robot.target_actor.state_dict(), 'src/tugbot_ml_controller/models/weights/best_actor_t.pth')
            torch.save(robot.target_critic.state_dict(), 'src/tugbot_ml_controller/models/weights/best_critic_t.pth')
            prev_score = score
        
        print("Score of " + str(score) + ", Episode: " + str(i))
        print("Noise Value: " + str(robot.expl_noise))    
        
        torch.save(robot.local_actor.state_dict(), 'src/tugbot_ml_controller/models/weights/checkpoint_actor.pth')
        torch.save(robot.local_critic.state_dict(), 'src/tugbot_ml_controller/models/weights/checkpoint_critic.pth')
        torch.save(robot.target_actor.state_dict(), 'src/tugbot_ml_controller/models/weights/checkpoint_actor_t.pth')
        torch.save(robot.target_critic.state_dict(), 'src/tugbot_ml_controller/models/weights/checkpoint_critic_t.pth')

    data = robot.memory.buffer
    fieldnames = ["State", "Action", "Reward","Next State","Complete"]
    with open('src/tugbot_ml_controller/CSV_Files/Buffer.csv','w', newline='') as Buf:
        writer = csv.writer(Buf)
        writer.writerow(fieldnames)
        writer.writerows(data)


    print('Training Saved')
    return list_rewards, list_time, robot.loss_a, robot.loss_c

def data_manip(data: str, row):
    if row == 0:
        data = data.replace('\n','')
        data = data.replace('[','')
        data = data.replace(']','')
        array_data = np.fromstring(data, sep=' ')
    elif row == 1 or row == 3:
        data = data.replace('[','')
        data = data.replace(']','')
        array_data = np.fromstring(data, sep=',')
    elif row == 5:
        data = data.replace('[','')
        data = data.replace(']','')
        array_data = np.fromstring(data, sep=' ')
    else:
        data = data.replace('\n','')
        data = data.replace('[','')
        data = data.replace(']','')
        array_data = np.fromstring(data, sep=',')

    return array_data

def test():
    robot = gazebo_Interface()

    robot.local_actor.load_state_dict(torch.load('src/tugbot_ml_controller/models/weights/checkpoint_actor_Test7.pth', map_location="cpu"))
    robot.target_actor.load_state_dict(torch.load('src/tugbot_ml_controller/models/weights/checkpoint_actor_t_Test7.pth', map_location="cpu"))
    robot.local_critic.load_state_dict(torch.load('src/tugbot_ml_controller/models/weights/checkpoint_critic_Test7.pth', map_location="cpu"))
    robot.target_critic.load_state_dict(torch.load('src/tugbot_ml_controller/models/weights/checkpoint_critic_t_Test7.pth', map_location="cpu"))

    complete = False
    state = robot.call_reset(robot,1,550)
    max_timesteps = 550
    timestep = 0
    torch.manual_seed(0)
    np.random.seed(0) 

    while True:
        action = robot.act(state)
        action = [(action[0] + 1)/2, action[1]]
        next_state, reward, complete = robot.step(action,robot)
        
        if complete:
            state = robot.call_reset(robot,16)
            complete = False
            timestep = 0
        else:
            state = next_state
            timestep += 0

def main(args=None):
    rclpy.init()
    #test()
    
    rewards, time, loss_a, loss_c = train(10, 550, pretrained=True,New_run=False)#always check all the parameters
    
    loss_a=[tensor.detach().numpy() for tensor in loss_a]
    loss_c=[tensor.detach().numpy() for tensor in loss_c]

    fig = plt.figure()
    plt.plot(np.arange(1,len(rewards)+1), rewards)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    fig = plt.figure()
    plt.plot(np.arange(1,len(time)+1), time)
    plt.ylabel('Time')
    plt.xlabel('Episode #')
    plt.show()

    fig = plt.figure()
    plt.plot(np.arange(1,len(loss_a)+1), loss_a) 
    plt.ylabel('Loss Actor')
    plt.xlabel('Episode #')
    plt.show()

    fig = plt.figure()
    plt.plot(np.arange(1,len(loss_c)+1), loss_c)
    plt.ylabel('Loss Critic')
    plt.xlabel('Episode #')
    plt.show()
    
    rclpy.shutdown()
    

if __name__=='__main__':
    main()



















