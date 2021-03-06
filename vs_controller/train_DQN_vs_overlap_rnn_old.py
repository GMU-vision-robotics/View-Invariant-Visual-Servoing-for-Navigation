import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import math
from math import sin, cos, pi, sqrt, atan2
import glob
import sys
import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/rrt')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/DDPG')
from util_visual_servoing import get_train_test_scenes, get_mapper, get_mapper_scene2points, create_folder, get_mapper_dist_theta_heading, get_pose_from_name, sample_gt_random_dense_correspondences, sample_gt_dense_correspondences
import rrt
import random
from math import cos, sin, pi
from util import action2pose, func_pose2posAndorn, similar_location_under_certainThreshold, plus_theta_fn, minus_theta_fn
from util_vscontroller import gt_goToPose, genOverlapAreaOnCurrentView, genOverlapAreaOnGoalView, genErrorOverlapAreaOnCurrentView
from utils_ddpg import buildActionMapper, update_current_pose
from dqn_vs import DQN_vs_overlap_rnn
import torch

np.set_printoptions(precision=2, suppress=True)

## necessary constants
mapper_scene2points = get_mapper_scene2points()
num_episodes = 200000
batch_size = 1
lambda_action = 0.25
action_table = buildActionMapper(flag_fewer_actions=True)
seq_len = 50
distance_metric = 'goToPose'

#def main(scene_idx=0, actual_episodes=1):

scene_idx = 0
actual_episodes=1

Train_Scenes, Test_Scenes = get_train_test_scenes()
scene_name = Train_Scenes[scene_idx]
num_startPoints = len(mapper_scene2points[scene_name])
model_weights_save_path = '{}'.format('/home/reza/Datasets/GibsonEnv/my_code/vs_controller/trained_dqn')
action_space = action_table.shape[0]

##=============================================================================================================
## rrt functions
## first figure out how to sample points from rrt graph
rrt_directory = '/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt'.format(scene_name)
path_finder = rrt.PathFinder(rrt_directory)
path_finder.load()
num_nodes = len(path_finder.nodes_x)
free = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt/free.png'.format(scene_name), 0)

##------------------------------------------------------------------------------------------------------------
## setup environment
import gym, logging
from mpi4py import MPI
from gibson.envs.husky_env import HuskyNavigateEnv
from baselines import logger
import skimage.io
from transforms3d.euler import euler2quat
config_file = os.path.join('/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop', 'env_yamls', '{}_navigate.yaml'.format(scene_name))
env = HuskyNavigateEnv(config=config_file, gpu_count = 1)
obs = env.reset() ## this line is important otherwise there will be an error like 'AttributeError: 'HuskyNavigateEnv' object has no attribute 'potential''
mapper_scene2z = get_mapper()

def get_obs(current_pose):
	pos, orn = func_pose2posAndorn(current_pose, mapper_scene2z[scene_name])
	env.robot.reset_new_pose(pos, orn)
	obs, _, _, _ = env.step(4)
	obs_rgb = obs['rgb_filled']
	obs_depth = obs['depth']
	#obs_normal = obs['normal']
	return obs_rgb, obs_depth#, obs_normal

def close_to_goal(pose1, pose2, thresh=0.15):
	L2_dist = math.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)
	thresh_L2_dist = thresh
	theta_change = abs(pose1[2] - pose2[2])/math.pi * 180
	return (L2_dist < thresh_L2_dist) and (theta_change <= 30)

def compute_distance_old(left_pose, right_pose, lamb=0.5):
	x1, y1 = left_pose[0], left_pose[1]
	a1, b1 = cos(left_pose[2]), sin(left_pose[2])
	x2, y2 = right_pose[0], right_pose[1]
	a2, b2 = cos(right_pose[2]), sin(right_pose[2])
	x_y_dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
	theta_dist = math.sqrt((a1-a2)**2 + (b1-b2)**2)
	return  x_y_dist + lamb * theta_dist

def compute_distance(left_pose, right_pose, lamb_alpha=0.5, lamb_beta=0.2):
	x1, y1 = left_pose[0], left_pose[1]
	x2, y2 = right_pose[0], right_pose[1]
	pho_dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
	
	left_pose_heading = left_pose[2]
	right_pose_heading = right_pose[2]
	location_angle = atan2(y2-y1, x2-x1)
	#print('left_pose_heading = {}, right_pose_heading = {}, location_angle = {}'.format(left_pose_heading, right_pose_heading, location_angle))
	if pho_dist >= 0.05:
		## alpha angle in goToPose is the difference between location angle and left_pose_heading
		a1, b1 = cos(location_angle), sin(location_angle)
		a2, b2 = cos(left_pose_heading), sin(left_pose_heading)
		alpha_dist = math.sqrt((a1-a2)**2 + (b1-b2)**2)
		## beta angle in goToPose is the difference between right_pose_heading and location angle
		a1, b1 = cos(right_pose_heading), sin(right_pose_heading)
		a2, b2 = cos(location_angle), sin(location_angle)
		beta_dist = math.sqrt((a1-a2)**2 + (b1-b2)**2)
	else:
		## when pho_dist is close to zero, alpha_dist is not important
		alpha_dist = 0.0
		## beta angle becomes the anlge between left and right poses
		a1, b1 = cos(right_pose_heading), sin(right_pose_heading)
		a2, b2 = cos(left_pose_heading), sin(left_pose_heading)
		beta_dist = math.sqrt((a1-a2)**2 + (b1-b2)**2)
	#print('pho_dist = {:.2f}, alpha_dist = {:.2f}, beta_dist = {:.2f}'.format(pho_dist, alpha_dist, beta_dist))
	return  pho_dist + lamb_alpha * alpha_dist + lamb_beta * beta_dist

def decide_reward_and_done(previous_pose, current_pose, goal_pose, start_pose):
	## check if the new step is on free space or not
	reward = 0.0
	done = 0
	
	## check if current_pose is closer to goal_pose than previous_pose
	'''
	L2_dist_current = math.sqrt((current_pose[0] - goal_pose[0])**2 + (current_pose[1] - goal_pose[1])**2)
	L2_dist_previous = math.sqrt((previous_pose[0] - goal_pose[0])**2 + (previous_pose[1] - goal_pose[1])**2)
	if L2_dist_current < L2_dist_previous:
		reward += 0.25
	print('L2_dist_current = {:.2f}, L2_dist_previous = {:.2f}, reward = {}'.format(L2_dist_current, L2_dist_previous, reward))
	'''

	## following Fereshteh's DiVIs paper
	dist_init = compute_distance(start_pose, goal_pose, lamb_alpha=0.2)
	dist_current = compute_distance(current_pose, goal_pose, lamb_alpha=0.2)
	reward = max(0, 1 - min(dist_init, dist_current)/(dist_init+0.00001))
	#print('dist_init = {:.2f}, dist_current = {:.2f}, reward = {:.2f}'.format(dist_init, dist_current, reward))
	
	## check if current_pose is close to goal
	## goal reward should be larger than all the previously accumulated reward
	flag_close_to_goal = close_to_goal(current_pose, goal_pose)
	if flag_close_to_goal:
		reward = 50.0
		done = 1
	#print('current_pose = {}, goal_pose = {}, flag_close_to_goal = {}, reward = {}'.format(current_pose, goal_pose, flag_close_to_goal, reward))

	#collision_done = 0
	## if there is a collision, reward is -1 and the episode is done
	left_pixel = path_finder.point_to_pixel((previous_pose[0], previous_pose[1]))
	right_pixel = path_finder.point_to_pixel((current_pose[0], current_pose[1]))
	## rrt.line_check returns True when there is no obstacle
	if not rrt.line_check(left_pixel, right_pixel, free):
		print('bumped into obstacle ....')
		reward = 0.0
		#collision_done = 1
		done=1
	print('final reward = {}'.format(reward))
	
	return reward, done, 0 #, collision_done

##============================================================================================================
base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_{}'.format('train')

agent = DQN_vs_overlap_rnn(trained_model_path=None, num_actions=action_space, input_channels=2)
#agent = DQN_vs_overlap_rnn(trained_model_path=model_weights_save_path, num_actions=action_space, input_channels=2)

rewards = []
avg_rewards = []

for i_epoch in range(actual_episodes):
	## go through each point folder
	for point_idx in range(0, num_startPoints):
	#for point_idx in range(0, 1):
		print('point_idx = {}'.format(point_idx))

		## read in start img and start pose
		point_image_folder = '{}/{}/point_{}'.format(base_folder, scene_name, point_idx)
		point_pose_npy_file = np.load('{}/{}/point_{}_poses.npy'.format(base_folder, scene_name, point_idx))

		#start_img = cv2.imread('{}/{}.png'.format(point_image_folder, point_pose_npy_file[0]['img_name']))[:, :, ::-1]
		start_pose = point_pose_npy_file[0]['pose']
		start_img, start_depth = get_obs(start_pose)
		start_depth = start_depth.copy()

		## index 0 is the left image, so right_img_idx starts from index 1
		for right_img_idx in range(1, len(point_pose_npy_file)):
		#for right_img_idx in range(3, 4):
			#print('right_img_idx = {}'.format(right_img_idx))

			current_pose = start_pose
			
			right_img_name = point_pose_npy_file[right_img_idx]['img_name']
			goal_pose = point_pose_npy_file[right_img_idx]['pose']
			#goal_img = cv2.imread('{}/{}.png'.format(point_image_folder, right_img_name), 1)[:,:,::-1]
			goal_img, goal_depth = get_obs(goal_pose)
			goal_img, goal_depth = goal_img.copy(), goal_depth.copy()

			overlapArea = genOverlapAreaOnCurrentView(start_depth, goal_depth, start_pose, goal_pose)[:,:,:2]
			state = [overlapArea]

			episode_reward = 0

			local_memory = []

			hidden_state, cell_state = agent.actor.init_hidden_states(batch_size=1)

			for i_step in range(seq_len):
				action, hidden_state, cell_state = agent.select_action(state, hidden_state, cell_state)
				print('action = {}'.format(action))
				
				## update current_pose
				vz, omegay = action_table[action]
				#print('vz = {:.2f}, omegay = {:.2f}'.format(vz, omegay))
				vx = 0.0
				vx = vx * lambda_action
				vz = vz * lambda_action
				omegay = omegay * pi * lambda_action
				#print('actual velocity = {:.2f}, {:.2f}, {:.2f}'.format(vx, vz, omegay))
				previous_pose = current_pose
				current_pose = update_current_pose(current_pose, vx, vz, omegay)
				## compute new_state
				current_img, current_depth = get_obs(current_pose)
				next_left_img, next_left_depth = current_img.copy(), current_depth.copy()

				new_overlapArea = genOverlapAreaOnCurrentView(next_left_depth, goal_depth, current_pose, goal_pose)[:,:,:2]
				new_state = [new_overlapArea]

				## visualize the state
				
				## collision done only stops continuing the sequence, but won't affect reward computing
				reward, done, collision_done = decide_reward_and_done(previous_pose, current_pose, goal_pose, start_pose)
				print('done = {}, collision_done = {}'.format(done, collision_done))
				if i_step == seq_len-1:
					print('used up all the steps ...')
					done = 1

				local_memory.append((state, action, reward, new_state, done))
				
				if len(agent.memory) >=2:
					if len(agent.memory) > 16:
						agent.update(16)
					else:
						agent.update(2)

				state = new_state
				episode_reward += reward
				print('---------------- end of a action ------------------ ')

				if done or collision_done:
					break

			print('---------------- end of a sequence ------------------ ')
			agent.memory.add_episode(local_memory)

			rewards.append(episode_reward)
			avg_rewards.append(np.mean(rewards[-10:]))
			sys.stdout.write("------------------------------------epoch = {}, point = {}, traj = {}, reward: {}, average_reward: {} #_steps: {}\n".format(i_epoch, point_idx, right_img_idx, np.round(episode_reward, decimals=2), np.round(avg_rewards[-1], decimals=2), i_step))

			if right_img_idx % 10 == 0:
				agent.update_critic()

				## plot the running_loss
				plt.plot(rewards, label='reward')
				plt.plot(avg_rewards, label='avg_reward')
				plt.xlabel('Episode')
				plt.ylabel('Reward')
				plt.grid(True)
				plt.legend(loc='upper right')
				plt.yscale('linear')
				plt.title('change of reward and avg_reward')
				plt.savefig('{}/Reward_episode_{}_{}.jpg'.format(
					model_weights_save_path, num_episodes, scene_name), bbox_inches='tight')
				plt.close()

				torch.save(agent.actor.state_dict(), '{}/dqn_epoch_200000_{}.pt'.format(model_weights_save_path, scene_name))
				torch.save(agent.actor.state_dict(), '{}/dqn_epoch_200000.pt'.format(model_weights_save_path))

'''
if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--scene_idx', type=int, default=0)
	args = parser.parse_args()
	main(args.scene_idx)
'''

class DQN_vs_overlap_rnn:
    def __init__(self, trained_model_path=None, num_actions=2, input_channels=2, actor_learning_rate=1e-5, critic_learning_rate=1e-4, gamma=0.97, tau=1e-2, max_memory_size=1000):
        # Params
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.steps_done = 0
        self.input_channels = input_channels

        # Networks
        self.perception_actor = Perception_overlap_recurrent(input_channels).to(device)
        self.actor = DQN_OVERLAP_Recurrent_Controller(self.perception_actor, self.num_actions, input_size=256).to(device)
        self.perception_critic = Perception_overlap_recurrent(input_channels).to(device)
        self.critic = DQN_OVERLAP_Recurrent_Controller(self.perception_critic, self.num_actions, input_size=256).to(device)

        if trained_model_path != None:
            self.actor.load_state_dict(torch.load('{}/dqn_epoch_200000.pt'.format(trained_model_path)))
            print('*********************************successfully read the model ...')

        ## copy params from actor.parameters to critic.parameters.
        ## when calling the copy_(),  the argument param.data is the src.
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()

        # Training
        self.memory = ReplayMemory_overlap_dqn_recurrent(max_memory_size)        
        ## only update weights of actor's linear layer
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=actor_learning_rate)

    def update_critic(self):
        for target_param, param in zip(self.critic.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic.eval()
    
    ## for collecting (state, action, next_state) tuples
    def select_action(self, state, hidden_state, cell_state, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=10000):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        ## take action with the maximum reward by using the actor
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                obs = state[0] ## 256 x 256 x 1
                print('obs.shape = {}'.format(obs.shape))
                #print('obs.shape = {}'.format(obs.shape))
                obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2)
                model_out = self.actor.forward(obs, hidden_state, cell_state, batch_size=1, time_step=1)
                action = model_out[0].max(1)[1].view(1, 1)
                hidden_state = model_out[1][0]
                cell_state = model_out[1][1]
                return action, hidden_state, cell_state
        else:
            ## take random actions, do exploration
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                obs = state[0] ## 256 x 256 x 1
                print('obs.shape = {}'.format(obs.shape))
                obs = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0)
                print('obs.shape = {}'.format(obs.shape))
                model_out = self.actor.forward(obs, hidden_state, cell_state, batch_size=1, time_step=1)
                action = torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)
                hidden_state = model_out[1][0]
                cell_state = model_out[1][1]
                return action, hidden_state, cell_state
    
    def update(self, batch_size):
        loss = 0.0
        for _ in range(batch_size):
            hidden_batch, cell_batch = self.actor.init_hidden_states(batch_size=1)
            episode = self.memory.sample(1, time_step=5)[0]
            len_episode = len(episode)
            #print('len_episode = {}'.format(len_episode))
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            for elem in episode:
                state, action, reward, next_state, done = elem
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

            actions = torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(0)
            print('actions.shape = {}'.format(actions.shape))
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(0)
            dones = torch.tensor(dones, dtype=torch.float32).to(device).unsqueeze(1).unsqueeze(0)
            states = torch.tensor(states, dtype=torch.float32).to(device).squeeze(1).permute(0, 3, 1, 2).unsqueeze(0)
            next_states = torch.tensor(next_states, dtype=torch.float32).to(device).squeeze(1).permute(0, 3, 1, 2).unsqueeze(0)
            
            # Critic loss (value function loss)  
            ## Get predicted next-state actions and Q values from target models
            ## gather() accumulates values at given index
            Qvals, _ = self.actor.forward(states, hidden_batch, cell_batch, batch_size=1, time_step=len_episode)
            Qvals = Qvals[0, actions[0, len_episode-1]]

            next_Q = self.critic.forward(next_states, hidden_batch, cell_batch, batch_size=1, time_step=len_episode).max(1)[0].detach().unsqueeze(1)
            # Compute Q targets for current states (y_i)
            Qprime = rewards[0, len_episode-1] + self.gamma * next_Q * (1-done)
            #print('Qprime = {}'.format(Qprime))
            episode_loss = F.smooth_l1_loss(Qvals, Qprime)
            loss += episode_loss
            #print('loss = {}'.format(loss))
        
        # update networks
        self.actor_optimizer.zero_grad()
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()