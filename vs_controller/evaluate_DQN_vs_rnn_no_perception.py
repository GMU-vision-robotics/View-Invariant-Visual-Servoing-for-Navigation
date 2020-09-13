import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import math
from math import sin, cos, pi, sqrt
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
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/short_range_nav_comparison')
from util_visual_servoing import get_train_test_scenes, get_mapper, get_mapper_scene2points, create_folder, get_mapper_dist_theta_heading, get_pose_from_name, sample_gt_random_dense_correspondences, sample_gt_dense_correspondences
import rrt
import random
from math import cos, sin, pi
from util import action2pose, func_pose2posAndorn, similar_location_under_certainThreshold, plus_theta_fn, minus_theta_fn
from util_vscontroller import gt_goToPose, genOverlapAreaOnCurrentView, genOverlapAreaOnGoalView, genErrorOverlapAreaOnCurrentView, genGtDenseCorrespondenseFlowMap, normalize_opticalFlow, normalize_depth, genGtDenseCorrespondenseFlowMapAndMask, removeCorrespondenceRandomly_withSmoothing
from utils_ddpg import buildActionMapper, update_current_pose
from dqn_vs import DQN_vs_overlap, DQN_vs_siamese
from model_vs import *
import torch
from util_short_range import findShortRangeImageName

np.set_printoptions(precision=2, suppress=True)
approach = 'thirtyfourth_try_rnn_varyingLengthEpisode_FeedbackReward' #'twentyeighth_try_vanillaRNN_model' #'twentysixth_try_opticalFlow_rnn' 
mode = 'Test'
input_type = 'optical_flow_rnn'
scene_idx = 0

## necessary constants
mapper_scene2points = get_mapper_scene2points()
num_episodes = 200000
batch_size = 64
lambda_action = 0.25
action_table = buildActionMapper(flag_fewer_actions=True)
seq_len = 50

Train_Scenes, Test_Scenes = get_train_test_scenes()

if mode == 'Test':
	scene_name = Test_Scenes[scene_idx]
elif mode == 'Train':
	scene_name = Train_Scenes[scene_idx]

num_startPoints = len(mapper_scene2points[scene_name])
model_weights_save_path = '{}/{}'.format('/home/reza/Datasets/GibsonEnv/my_code/vs_controller/trained_dqn', approach)
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

def close_to_goal(pose1, pose2, thresh=0.20):
	L2_dist = math.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2)
	thresh_L2_dist = thresh
	theta_change = abs(pose1[2] - pose2[2])/math.pi * 180
	return (L2_dist <= thresh_L2_dist) #and (theta_change <= 30)

##============================================================================================================
if mode == 'Test':
	base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_{}'.format('test')
elif mode == 'Train':
	base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_{}'.format('train')

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0')     ## Default CUDA device
num_epochs = 200000 ## same as # of trajs sampled
num_actions = action_table.shape[0]
perception = Perception_overlap(2).to(device)
perception_model = DQN_OVERLAP_Controller(perception, action_space, input_size=256).to(device)
approach = 'twentyseventh_try_opticalFlow_newDistMetric'
perception_model.load_state_dict(torch.load('{}/{}/dqn_epoch_{}_Uvalda.pt'.format('/home/reza/Datasets/GibsonEnv/my_code/vs_controller/trained_dqn', approach, 200000)))

model = DQN_OVERLAP_Recurrent_Controller_no_perception_dense(num_actions, input_size=256).to(device)
model.load_state_dict(torch.load('{}/dqn_epoch_{}_Uvalda.pt'.format(model_weights_save_path, num_epochs)))

list_succ = []
list_collision = []
## go through each point folder
if mode == 'Test':
	a, b = 0, 1
elif mode == 'Train':
	a, b = 7, 8
	#a, b = 0, 1
#for point_idx in range(0, num_startPoints):
for point_idx in range(a, b):
#for point_idx in range(6, 7):
	print('point_idx = {}'.format(point_idx))

	## read in start img and start pose
	point_image_folder = '{}/{}/point_{}'.format(base_folder, scene_name, point_idx)
	point_pose_npy_file = np.load('{}/{}/point_{}_poses.npy'.format(base_folder, scene_name, point_idx))

	#start_img = cv2.imread('{}/{}.png'.format(point_image_folder, point_pose_npy_file[0]['img_name']))[:, :, ::-1]
	start_pose = point_pose_npy_file[0]['pose']
	start_img, start_depth = get_obs(start_pose)
	start_depth = start_depth.copy()

	count_succ = 0
	count_collision = 0
	count_short_runs = 0
	count_short_runs_collision = 0
	count_short_runs_succ = 0
	## index 0 is the left image, so right_img_idx starts from index 1
	for right_img_idx in range(1, len(point_pose_npy_file)):
	#for right_img_idx in range(3, 4):
		print('right_img_idx = {}'.format(right_img_idx))

		current_pose = start_pose
		
		right_img_name = point_pose_npy_file[right_img_idx]['img_name']
		
		goal_pose = point_pose_npy_file[right_img_idx]['pose']
		#goal_img = cv2.imread('{}/{}.png'.format(point_image_folder, right_img_name), 1)[:,:,::-1]
		goal_img, goal_depth = get_obs(goal_pose)
		goal_depth = goal_depth.copy()

		current_depth = start_depth.copy()

		episode_reward = 0

		flag_succ = False

		#hidden_state, cell_state = model.init_hidden_states(batch_size=1)
		hidden_state= model.init_hidden_states(batch_size=1)
		for i_step in range(seq_len):
			overlapArea = genGtDenseCorrespondenseFlowMap(current_depth, goal_depth, current_pose, goal_pose)[:,:,:2]
			#overlapArea = removeCorrespondenceRandomly_withSmoothing(overlapArea, keep_prob=1.0)
			tensor_start = torch.tensor(overlapArea, dtype=torch.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2)
			ft_overlapArea = perception_model.perception(tensor_start).detach().cpu().numpy().squeeze(0)
			
			tensor_left = torch.tensor(ft_overlapArea, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(1)
			#Qvalue_table, (hidden_state, cell_state) = model(tensor_left, hidden_state, cell_state, batch_size=1, time_step=1)
			Qvalue_table, hidden_state = model(tensor_left, hidden_state, batch_size=1)
			#hidden_state= model.init_hidden_states(batch_size=1)
			pred = Qvalue_table.squeeze(1).max(1)[1].view(1, 1).detach().cpu().numpy().item() ## batch_size x 3
			#print('Qvalue_table: {}'.format(Qvalue_table))
			#print('pred = {}'.format(pred))
			
			## update current_pose
			vz, omegay = action_table[pred]
			#print('vz = {:.2f}, omegay = {:.2f}'.format(vz, omegay))
			vx = 0.0
			vx = vx * lambda_action
			vz = vz * lambda_action
			omegay = omegay * pi * lambda_action
			#print('actual velocity = {:.2f}, {:.2f}, {:.2f}'.format(vx, vz, omegay))
			previous_pose = current_pose
			current_pose = update_current_pose(current_pose, vx, vz, omegay)

			flag_broken = False
			left_pixel = path_finder.point_to_pixel((previous_pose[0], previous_pose[1]))
			right_pixel = path_finder.point_to_pixel((current_pose[0], current_pose[1]))
			## rrt.line_check returns True when there is no obstacle
			if not rrt.line_check(left_pixel, right_pixel, free):
				print('run into something')
				flag_broken = True
				break

			if close_to_goal(current_pose, goal_pose):
				print('success run')
				flag_succ = True
				break

			## compute new_state
			current_img, current_depth = get_obs(current_pose)
			current_depth = current_depth.copy()

		if flag_succ:
			count_succ += 1
			list_succ.append(point_pose_npy_file[right_img_idx]['img_name'])
			if findShortRangeImageName(right_img_name):
				count_short_runs_succ += 1
		if flag_broken:
			count_collision += 1
			list_collision.append(point_pose_npy_file[right_img_idx]['img_name'])
			if findShortRangeImageName(right_img_name):
				count_short_runs_collision += 1
		if findShortRangeImageName(right_img_name):
			count_short_runs += 1


		print('count_succ = {}'.format(count_succ))
		print('count_collision = {}'.format(count_collision))
		print('count_short_runs_succ = {}'.format(count_short_runs_succ))
		print('count_short_runs_collision = {}'.format(count_short_runs_collision))

	print('num_succ = {}, num_run = {}, count_short_runs_succ = {}, count_short_runs = {}'.format(count_succ, len(point_pose_npy_file), count_short_runs_succ, count_short_runs))

'''
f = open('{}/successful_runs_{}.txt'.format('/home/reza/Datasets/GibsonEnv/my_code/vs_controller', approach), 'w')
for i in range(len(list_succ)):
	f.write('{}\n'.format(list_succ[i]))
f.close()
'''

