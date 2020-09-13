import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import math
from math import sin, cos
import glob
import sys
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/rrt')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/keypointNet')
import rrt
import random
from math import cos, sin, pi
from util import action2pose, func_pose2posAndorn, similar_location_under_certainThreshold, plus_theta_fn, minus_theta_fn
from util_visual_servoing import get_train_test_scenes, get_mapper, get_mapper_scene2points, create_folder, compute_gt_velocity_through_interaction_matrix, sample_gt_random_dense_correspondences, compute_velocity_through_correspondences_and_depth
from util_vscontroller import genGtRandomDenseCorrespondenseFlowMap
from utils_keypointNet import detect_learned_correspondences

scene_idx = 0

mapper_scene2z = get_mapper()
mapper_scene2points = get_mapper_scene2points()
Train_Scenes, Test_Scenes = get_train_test_scenes()

#def main(scene_idx):

scene_name = Test_Scenes[scene_idx]
num_startPoints = len(mapper_scene2points[scene_name])

print('scene_name = {}'.format(scene_name))

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
	obs_depth = obs['depth']
	return obs_rgb, obs_depth


## create folder for loading the training images
base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_test'
## create folder for saving the collected replay memory
#approach_folder = '{}/gtDenseCorrespondence_interactionMatrixFromLargeDisplacementGtCorrespondence'.format('/home/reza/Datasets/GibsonEnv/my_code/vs_controller/replayMemory_supervisedLearning')
approach_folder = '{}/learnedCorrespondence_interactionMatrixFromLearnedCorrespondence'.format('/home/reza/Datasets/GibsonEnv/my_code/vs_controller/replayMemory_supervisedLearning')
create_folder(approach_folder)

for point_idx in range(1, num_startPoints):
	print('point_idx = {}'.format(point_idx))
	list_trajs = []

	## read in start img and start pose
	point_image_folder = '{}/{}/point_{}'.format(base_folder, scene_name, point_idx)
	point_pose_npy_file = np.load('{}/{}/point_{}_poses.npy'.format(base_folder, scene_name, point_idx))

	start_img = cv2.imread('{}/{}.png'.format(point_image_folder, point_pose_npy_file[0]['img_name']))[:, :, ::-1]
	start_pose = point_pose_npy_file[0]['pose']

	## index 0 is the left image, so right_img_idx starts from index 1
	for right_img_idx in range(1, len(point_pose_npy_file)):
		flag_correct = False
		print('right_img_idx = {}'.format(right_img_idx))
		seq_len = 50
		count_steps = 0
		list_actions = []
		flag_broken = False

		current_img = start_img.copy()
		current_pose = [start_pose[0], start_pose[1], start_pose[2]]
		
		right_img_name = point_pose_npy_file[right_img_idx]['img_name']
		goal_img = cv2.imread('{}/{}.png'.format(point_image_folder, right_img_name), 1)[:,:,::-1]
		goal_depth = np.load('{}/{}_depth.npy'.format(point_image_folder, right_img_name))
		goal_pose = point_pose_npy_file[right_img_idx]['pose']

		#list_states = np.zeros((seq_len+1, 256, 256, 4), dtype=np.float32) ## num_steps x 256 x 256 x 4
		list_actions = np.zeros((seq_len+1, 3), dtype=np.float32) ## num_steps x 3
		list_kp1 = []
		list_kp2 = []
		list_depth = np.zeros((seq_len+1, 256, 256), dtype=np.float32)
		
		while count_steps < seq_len:
			current_img, current_depth = get_obs(current_pose)
			## generate random dense correspondence 
			#kp1, kp2 = sample_gt_random_dense_correspondences(current_depth, goal_depth, current_pose, goal_pose)
			
			## generate learned correspondence
			kp1, kp2 = detect_learned_correspondences(current_img, goal_img)

			## compute gt velocity and decide stop or not
			#vx, vz, omegay, flag_stop = compute_gt_velocity_through_interaction_matrix(current_depth, goal_depth, current_pose, goal_pose)

			vx, vz, omegay, flag_stop = compute_velocity_through_correspondences_and_depth(kp1, kp2, current_depth)

			## build states and actions
			#list_states[count_steps] = np.stack((y_flow, x_flow, mask, np.squeeze(current_depth, axis=-1)), axis=-1)
			list_kp1.append(kp1)
			list_kp2.append(kp2)
			list_depth[count_steps] = np.squeeze(current_depth, axis=-1)
			if flag_stop:
				list_actions[count_steps] = [0.0, 0.0, 0.0]
			else:
				list_actions[count_steps] = [vx, vz, omegay]

			## compute next pose
			x, y, theta = current_pose
			vx_theta = minus_theta_fn(pi/2, theta)
			x = x + vz * cos(theta) + vx * cos(vx_theta)
			y = y + vz * sin(theta) + vx * sin(vx_theta)
			theta = theta + omegay
			current_pose = [x, y, theta]

			count_steps += 1

			## stop navigation
			if flag_stop:
				break

			## check if the new step is on free space or not
			if not path_finder.pc.free_point(current_pose[0], current_pose[1]):
				flag_broken = True
				break

		list_actions = list_actions[:count_steps]
		#list_states = list_states[:count_steps]
		list_depth = list_depth[:count_steps]

		list_trajs.append([list_kp1, list_kp2, list_depth, list_actions])

	np.save('{}/{}_point_{}.npy'.format(approach_folder, scene_name, point_idx), list_trajs)

'''
if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--scene_idx', type=int, default=0)
	args = parser.parse_args()
	main(args.scene_idx)
'''