import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import math
from math import sin, cos
from util_visual_servoing import get_train_test_scenes, get_mapper, detect_correspondences, get_mapper_scene2points, create_folder, get_mapper_dist_theta_heading, get_pose_from_name, sample_gt_correspondences_with_large_displacement, build_L_matrix, compute_velocity_through_correspondences_and_depth, goToPose_one_step
import glob
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
from util import action2pose, func_pose2posAndorn, similar_location_under_certainThreshold, plus_theta_fn, minus_theta_fn

np.set_printoptions(precision=2, suppress=True)

#def main(scene_idx=0):

scene_idx = 0

mapper_scene2z = get_mapper()
mapper_scene2points = get_mapper_scene2points()
Train_Scenes, Test_Scenes = get_train_test_scenes()
scene_name = Test_Scenes[scene_idx]
num_startPoints = len(mapper_scene2points[scene_name])
num_steps = 50

## create test folder
test_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/test_IBVS'
approach_folder = '{}/gtCorrespondence_interMatrix_gtDepth_Vz_OmegaY'.format(test_folder)
create_folder(approach_folder)

scene_folder = '{}/{}'.format(approach_folder, scene_name)
create_folder(scene_folder)

## rrt functions
## first figure out how to sample points from rrt graph
rrt_directory = '/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt'.format(scene_name)
path_finder = rrt.PathFinder(rrt_directory)
path_finder.load()
num_nodes = len(path_finder.nodes_x)
free = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt/free.png'.format(scene_name), 0)

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
	return obs_rgb.copy(), obs_depth.copy()

base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_{}'.format('test')

## go through each point folder
#for point_idx in range(0, num_startPoints):
for point_idx in range(0, 1):
	print('point_idx = {}'.format(point_idx))

	point_folder = '{}/point_{}'.format(scene_folder, point_idx)
	create_folder(point_folder)

	## read in start img and start pose
	point_image_folder = '{}/{}/point_{}'.format(base_folder, scene_name, point_idx)
	point_pose_npy_file = np.load('{}/{}/point_{}_poses.npy'.format(base_folder, scene_name, point_idx))

	start_img = cv2.imread('{}/{}.png'.format(point_image_folder, point_pose_npy_file[0]['img_name']))[:, :, ::-1]
	start_pose = point_pose_npy_file[0]['pose']

	#for right_img_idx in range(1, len(point_pose_npy_file)):
	for right_img_idx in range(97, 98):
		flag_correct = False
		print('right_img_idx = {}'.format(right_img_idx))

		count_steps = 0

		current_pose = start_pose
		
		right_img_name = point_pose_npy_file[right_img_idx]['img_name']
		goal_pose = point_pose_npy_file[right_img_idx]['pose']
		goal_img, goal_depth = get_obs(goal_pose)

		list_result_poses = [current_pose]
		num_matches = 0
		flag_broken = False

		list_displacement = []
		list_velocity = []
		list_omega = []

		while count_steps < num_steps:
			current_img, current_depth = get_obs(current_pose)
			try:
				kp1, kp2 = sample_gt_correspondences_with_large_displacement(current_depth, goal_depth, current_pose, goal_pose)
				if count_steps == 0:
					start_depth = current_depth.copy()
			except:
				print('run into error')
				break
			num_matches = kp1.shape[1]

			vx, vz, omegay, flag_stop, displacement = compute_velocity_through_correspondences_and_depth(kp1, kp2, current_depth)

			previous_pose = current_pose
			current_pose, v, omega, flag_stop_goToPose = goToPose_one_step(current_pose, vx, vz, omegay)

			list_displacement.append(displacement)
			list_velocity.append(v)
			list_omega.append(omega)

			## check if there is collision during the action
			left_pixel = path_finder.point_to_pixel((previous_pose[0], previous_pose[1]))
			right_pixel = path_finder.point_to_pixel((current_pose[0], current_pose[1]))
			# rrt.line_check returns True when there is no obstacle
			if not rrt.line_check(left_pixel, right_pixel, free):
				flag_broken = True
				print('run into an obstacle ...')
				break

			## check if we should stop or not
			if flag_stop or flag_stop_goToPose:
				print('flag_stop = {}, flag_stop_goToPose = {}'.format(flag_stop, flag_stop_goToPose))
				print('break')
				break

			count_steps += 1
			list_result_poses.append(current_pose)
			## sample current_img again to save in list_obs
			current_img, current_depth = get_obs(current_pose)
		#assert 1==2
		## decide if this run is successful or not
		flag_correct, dist, theta_change = similar_location_under_certainThreshold(goal_pose, list_result_poses[count_steps])
		print('dist = {}, theta = {}'.format(dist, theta_change))
		#print('start_pose = {}, final_pose = {}, goal_pose = {}'.format(start_pose, list_result_poses[-1], goal_pose))

		if flag_correct: 
			str_succ = 'Success'
		else:
			str_succ = 'Failure'

		## ===================================================================================================================
		assert 1==2
		## plot the pose graph
		img_name = 'goTo_{}.jpg'.format(right_img_name[10:])
		print('img_name = {}'.format(img_name))

		## plot the poses
		t = np.arange(len(list_displacement))

		plt.subplot(3, 1, 1)
		plt.plot(t, list_displacement)
		plt.xlabel('time step')
		plt.ylabel('average displacement')
		plt.yscale('linear')
		plt.grid(True)

		plt.subplot(3, 1, 2)
		plt.plot(t, list_velocity)
		plt.xlabel('time step')
		plt.ylabel('forward velocity')
		plt.yscale('linear')
		plt.grid(True)

		plt.subplot(3, 1, 3)
		plt.plot(t, list_omega)
		plt.xlabel('time step')
		plt.ylabel('angular velocity')
		plt.yscale('linear')
		plt.grid(True)

		plt.savefig('{}'.format(img_name), bbox_inches='tight', dpi=(400))
		plt.close()

		## ======================================================================================================================






		

'''
if __name__ == "__main__":	
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--scene_idx', type=int, default=0)
	args = parser.parse_args()
	main(args.scene_idx)
'''