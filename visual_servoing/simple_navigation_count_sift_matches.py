import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from util_visual_servoing import get_number_of_good_matches, get_train_test_scenes, get_mapper, get_num_matches_and_write_image

import sys
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/rrt')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop')
import rrt
import random
from math import cos, sin, pi
from util import action2pose, func_pose2posAndorn

Train_Scenes, Test_Scenes = get_train_test_scenes()
scene_idx = 10
scene_name = Train_Scenes[6]
mapper_scene2z = get_mapper()

## rrt functions
## first figure out how to sample points from rrt graph
rrt_directory = '/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt'.format(scene_name)
path_finder = rrt.PathFinder(rrt_directory)
path_finder.load()
num_nodes = len(path_finder.nodes_x)
free = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt/free.png'.format(scene_name))

## GibsonEnv setup
## For Gibson Env
import gym, logging
from mpi4py import MPI
from gibson.envs.husky_env import HuskyNavigateEnv
from baselines import logger
import skimage.io
from transforms3d.euler import euler2quat
config_file = os.path.join('/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop', 'env_yamls', '{}_navigate.yaml'.format(scene_name))
env = HuskyNavigateEnv(config=config_file, gpu_count = 1)
obs = env.reset() ## this line is important otherwise there will be an error like 'AttributeError: 'HuskyNavigateEnv' object has no attribute 'potential''

def get_obs(current_pose):
	pos, orn = func_pose2posAndorn(current_pose, mapper_scene2z[scene_name])
	env.robot.reset_new_pose(pos, orn)
	obs, _, _, _ = env.step(4)
	obs_rgb = obs['rgb_filled']
	return obs_rgb

testing_image_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/testing_image_pairs'
pair_idx = 4

current_img = cv2.imread('{}/pair_{}_left.png'.format(testing_image_folder, pair_idx))[:,:,::-1]
goal_img = cv2.imread('{}/pair_{}_right.png'.format(testing_image_folder, pair_idx))[:,:,::-1]

presampled_poses = np.load('{}/traj_poses_{}.npy'.format(testing_image_folder, pair_idx))
current_pose = presampled_poses[0]
goal_pose = presampled_poses[-1]

seq_len = 20
count_steps = 0
list_result_poses = [current_pose]
list_actions = []

previous_num_matches = 0
flag_broken = False
action_list = [0, 2, 5, 6, 9]
while count_steps < seq_len:
	action_match_matrix = np.zeros((len(action_list), 2), np.int16)
	for idx, action in enumerate(action_list):
		new_pose = action2pose(current_pose, action)
		new_obs = get_obs(new_pose)
		#num_match = get_number_of_good_matches(new_obs, goal_img)
		img_name = '{}/{}/step_{}_action_{}.jpg'.format('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/test_vs', pair_idx, count_steps, action)
		num_match = get_num_matches_and_write_image(new_obs, goal_img, img_name)
		action_match_matrix[idx, 0] = action 
		action_match_matrix[idx, 1] = num_match

	## find the action with the maximum matches
	idx_maximum_match = np.argmax(action_match_matrix[:, 1])
	match_maximum = action_match_matrix[idx_maximum_match, 1]
	best_action = action_match_matrix[idx_maximum_match, 0]
	list_actions.append(best_action)
	print('action_match_matrix: {}'.format(action_match_matrix))
	print('action to take: {}'.format(best_action))

	current_pose = action2pose(current_pose, best_action)
	list_result_poses.append(current_pose)

	## check if new_pose good or not
	if not path_finder.pc.free_point(current_pose[0], current_pose[1]):
		flag_broken = True
		break
	## check if we should stop or not
	if match_maximum < previous_num_matches:
		break
	else:
		previous_num_matches = match_maximum
	count_steps += 1

## plot the pose graph
file_addr = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/test_vs'
img_name = '{}_pair_{}.jpg'.format(scene_name, pair_idx)
print('img_name = {}'.format(img_name))
## plot the poses
rows, cols, _ = free.shape
plt.imshow(free)
for n in range(len(presampled_poses)):
	pose = presampled_poses[n]
	x, y = path_finder.point_to_pixel((pose[0], pose[1]))
	theta = pose[2]
	plt.arrow(x, y, cos(theta), sin(theta), color='b', \
		overhang=1, head_width=0.1, head_length=0.15, width=0.001)

for m in range(len(list_result_poses)):
	pose = list_result_poses[m]
	x, y = path_finder.point_to_pixel((pose[0], pose[1]))
	theta = pose[2]
	plt.arrow(x, y, cos(theta), sin(theta), color='y', \
		overhang=1, head_width=0.1, head_length=0.15, width=0.001)

plt.axis([0, cols, 0, rows])
plt.xticks([])
plt.yticks([])
plt.savefig('{}/{}'.format(file_addr, img_name), bbox_inches='tight', dpi=(400))
plt.close()

np.save('{}/actions_pair_{}.npy'.format(file_addr, pair_idx), np.array(list_actions))