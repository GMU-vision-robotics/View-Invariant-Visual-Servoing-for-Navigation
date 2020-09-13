import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt, atan2
from pyquaternion import Quaternion
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
from util import action2pose, func_pose2posAndorn, similar_location_under_certainThreshold, plus_theta_fn, minus_theta_fn

import rrt
import cv2

def main(scene_idx=0, point_a=0, right_a=0):
	#scene_idx = 0

	Train_Scenes, Test_Scenes = get_train_test_scenes()
	mapper_scene2points = get_mapper_scene2points()
	scene_name = Test_Scenes[scene_idx]
	num_startPoints = len(mapper_scene2points[scene_name])

	## as the move forward distance = 0.1, assume velocity is 0.01, it needs 10 steps.
	def pose_interpolation(start_pose, end_pose, num=10, include_endpoint=False):
		x0, y0, theta0 = start_pose
		x1, y1, theta1 = end_pose
		x = np.linspace(x0, x1, num=num, endpoint=include_endpoint)
		y = np.linspace(y0, y1, num=num, endpoint=include_endpoint)
		## convert to quaternion
		q0 = Quaternion(axis=[0, -1, 0], angle=theta0)
		q1 = Quaternion(axis=[0, -1, 0], angle=theta1)
		pose_list = []
		v = np.array([1, 0, 0])
		for idx, q in enumerate(Quaternion.intermediates(q0, q1, num-1, include_endpoints=True)):
			if idx < num:
				e, d, f = q.rotate(v)
				theta = atan2(f, e)
				current_pose = [x[idx], y[idx], theta]
				pose_list.append(current_pose)
		return pose_list

	## rrt functions
	## first figure out how to sample points from rrt graph
	rrt_directory = '/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt'.format(scene_name)
	path_finder = rrt.PathFinder(rrt_directory)
	path_finder.load()
	num_nodes = len(path_finder.nodes_x)
	free = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt/free.png'.format(scene_name), 0)

	## draw the observations
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

	base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_{}'.format('test')

	#for point_idx in range(2, 3):
	for point_idx in range(point_a, point_a+1):
		print('point_idx = {}'.format(point_idx))

		## read in start img and start pose
		point_image_folder = '{}/{}/point_{}'.format(base_folder, scene_name, point_idx)
		point_pose_npy_file = np.load('{}/{}/point_{}_poses.npy'.format(base_folder, scene_name, point_idx))

		save_folder = '{}/{}/point_{}'.format('/home/reza/Datasets/GibsonEnv/my_code/vs_controller/for_video', scene_name, point_idx)

		## index 0 is the left image, so right_img_idx starts from index 1
		#for right_img_idx in range(1, len(point_pose_npy_file)):
		#for right_img_idx in range(1, 101):
		for right_img_idx in range(right_a, right_a+1):

			right_img_name = point_pose_npy_file[right_img_idx]['img_name']

			## Read in pose npy file generated from DQN
			dqn_pose_npy_file = np.load('{}/run_{}/{}_waypoint_pose_list.npy'.format(save_folder, right_img_idx, right_img_name[10:]))

			start_pose = dqn_pose_npy_file[0]
			goal_pose = dqn_pose_npy_file[1]
			dqn_pose_list = dqn_pose_npy_file[2]

			goal_img, goal_depth = get_obs(goal_pose)
			goal_img = goal_img[:, :, ::-1]
			cv2.imwrite('{}/run_{}/goal_img.jpg'.format(save_folder, right_img_idx), goal_img)

			interpolated_pose_list = []
			## build the subsequence
			len_dqn_pose_list = len(dqn_pose_list)

			for i in range(len_dqn_pose_list - 1):
				first_pose = dqn_pose_list[i]
				second_pose = dqn_pose_list[i+1]
				subseq_pose_list = pose_interpolation(first_pose, second_pose)
				interpolated_pose_list += subseq_pose_list
			interpolated_pose_list.append(dqn_pose_list[-1])

			img_name = 'goTo_{}.jpg'.format('current')
			print('img_name = {}'.format(img_name))

			## plot the poses
			free2 = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt/free.png'.format(scene_name), 1)
			rows, cols, _ = free2.shape
			plt.imshow(free2)

			for m in range(len(interpolated_pose_list)):
			#for m in range(0, 100, 5):
				pose = interpolated_pose_list[m]
				x, y = path_finder.point_to_pixel((pose[0], pose[1]))
				theta = pose[2]
				plt.arrow(x, y, cos(theta), sin(theta), color='y', \
					overhang=1, head_width=0.1, head_length=0.15, width=0.001)
			## draw goal pose
			x, y = path_finder.point_to_pixel((goal_pose[0], goal_pose[1]))
			theta = goal_pose[2]
			plt.arrow(x, y, cos(theta), sin(theta), color='r', \
					overhang=1, head_width=0.1, head_length=0.15, width=0.001)
			## draw start pose
			x, y = path_finder.point_to_pixel((start_pose[0], start_pose[1]))
			theta = start_pose[2]
			plt.arrow(x, y, cos(theta), sin(theta), color='b', \
					overhang=1, head_width=0.1, head_length=0.15, width=0.001)

			plt.axis([0, cols, 0, rows])
			plt.xticks([])
			plt.yticks([])
			#plt.title('{}, start point_{}, goal viewpoint {}, {}'.format(scene_name, point_idx, right_img_name[10:], str_succ))
			plt.savefig('{}/run_{}/{}'.format(save_folder, right_img_idx, 'overview.jpg'), bbox_inches='tight', dpi=(400))
			plt.close()

		#'''
		for i in range(len(interpolated_pose_list)):
			current_pose = interpolated_pose_list[i]
			obs_rgb, obs_depth = get_obs(current_pose)
			obs_rgb = obs_rgb[:, :, ::-1]
			cv2.imwrite('{}/run_{}/step_{}.jpg'.format(save_folder, right_img_idx, i), obs_rgb)
		#'''

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--scene_idx', type=int, default=0)
	parser.add_argument('--point_a', type=int, default=0)
	parser.add_argument('--right_a', type=int, default=0)
	args = parser.parse_args()
	main(args.scene_idx, args.point_a, args.right_a)