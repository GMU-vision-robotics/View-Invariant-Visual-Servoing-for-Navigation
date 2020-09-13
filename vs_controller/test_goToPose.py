import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import math
from math import sin, cos, sqrt, pi, atan2
import glob
import sys
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/rrt')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing')
import rrt
import random 
from util import action2pose, func_pose2posAndorn, similar_location_under_certainThreshold, plus_theta_fn, minus_theta_fn
from util_visual_servoing import get_train_test_scenes, get_mapper, get_mapper_scene2points, create_folder, compute_gt_velocity_through_interaction_matrix
from util_vscontroller import genOverlapAreaOnCurrentView, gt_goToPose, genOverlapAreaOnGoalView, genErrorOverlapAreaOnCurrentView, genErrorOverlapAreaOnGoalView

np.set_printoptions(precision=2, suppress=True)
#action_table = buildActionMapper(flag_fewer_actions=True)
mapper_scene2z = get_mapper()
mapper_scene2points = get_mapper_scene2points()
Train_Scenes, Test_Scenes = get_train_test_scenes()
num_steps = 50

#def main(scene_idx=0, category='Test'):

scene_idx = 0
category = 'Train'

if category == 'Test':
	scene_name = Test_Scenes[scene_idx]
if category == 'Train':
	scene_name = Train_Scenes[scene_idx]
num_startPoints = len(mapper_scene2points[scene_name])

print('scene_name = {}'.format(scene_name))

approach = 'third_try'
approach_folder = '/home/reza/Datasets/GibsonEnv/my_code/vs_controller/test_goToPose/{}'.format(approach)
test_folder = '{}/{}'.format(approach_folder, scene_name)
create_folder(test_folder)

## rrt functions
## first figure out how to sample points from rrt graph
rrt_directory = '/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt'.format(scene_name)
path_finder = rrt.PathFinder(rrt_directory)
path_finder.load()
num_nodes = len(path_finder.nodes_x)
free = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt/free.png'.format(scene_name), 0)

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

def compute_distance(left_pose, right_pose, lamb_alpha=0.5, lamb_beta=0.2):
	x1, y1 = left_pose[0], left_pose[1]
	x2, y2 = right_pose[0], right_pose[1]
	pho_dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
	
	left_pose_heading = left_pose[2]
	right_pose_heading = right_pose[2]
	location_angle = atan2(y2-y1, x2-x1)
	print('left_pose_heading = {}, right_pose_heading = {}, location_angle = {}'.format(left_pose_heading, right_pose_heading, location_angle))
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
	print('pho_dist = {:.2f}, alpha_dist = {:.2f}, beta_dist = {:.2f}'.format(pho_dist, alpha_dist, beta_dist))
	return  pho_dist + lamb_alpha * alpha_dist + lamb_beta * beta_dist

## create folder for loading the training images
if category == 'Test':
	base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_{}'.format('test')
elif category == 'Train':
	base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_{}'.format('train')

## go through each point folder
#for point_idx in range(0, num_startPoints):
for point_idx in range(7, 8):
	print('point_idx = {}'.format(point_idx))

	point_folder = '{}/point_{}'.format(test_folder, point_idx)
	create_folder(point_folder)

	## read in start img and start pose
	point_image_folder = '{}/{}/point_{}'.format(base_folder, scene_name, point_idx)
	point_pose_npy_file = np.load('{}/{}/point_{}_poses.npy'.format(base_folder, scene_name, point_idx))

	start_img = cv2.imread('{}/{}.png'.format(point_image_folder, point_pose_npy_file[0]['img_name']))[:, :, ::-1]
	start_pose = point_pose_npy_file[0]['pose']

	## index 0 is the left image, so right_img_idx starts from index 1
	count_correct = 0
	list_correct_img_names = []
	list_whole_stat = []

	#for right_img_idx in range(1, len(point_pose_npy_file)):
	for right_img_idx in range(1, 20):
		flag_correct = False
		print('============================================================================================')
		print('right_img_idx = {}'.format(right_img_idx))
		seq_len = 50
		count_steps = 0

		current_pose = start_pose
		
		right_img_name = point_pose_npy_file[right_img_idx]['img_name']
		goal_pose = point_pose_npy_file[right_img_idx]['pose']
		goal_img, goal_depth = get_obs(goal_pose)
		goal_img = goal_img.copy()
		goal_depth = goal_depth.copy()

		list_result_poses = [current_pose]

		gt_odometry, gt_actions = gt_goToPose(current_pose, goal_pose)
		#assert 1==2 
		num_actions = len(gt_actions)
		for i in range(num_actions):
			
			## update new pose in different ways depending on the chosen module
			previous_pose = gt_odometry[i]
			current_pose = gt_odometry[i+1]

			print('-------------------------------------------------------------------------------------------')
			print('left_pose = {}, right_pose = {}'.format(previous_pose, goal_pose))
			pose_dist = compute_distance(previous_pose, goal_pose, lamb_alpha=0.2)
			print('pose_dist = {}'.format(pose_dist))

			## check if there is collision during the action
			left_pixel = path_finder.point_to_pixel((previous_pose[0], previous_pose[1]))
			right_pixel = path_finder.point_to_pixel((current_pose[0], current_pose[1]))
			# rrt.line_check returns True when there is no obstacle
			if not rrt.line_check(left_pixel, right_pixel, free):
				flag_broken = True
				break

			## check the number of steps used
			list_result_poses.append(current_pose)
			count_steps += 1
			if count_steps >= num_steps:
				print('running out of steps')
				break

			'''
			previous_img, previous_depth = get_obs(previous_pose)
			previous_img = previous_img.copy()
			previous_depth = previous_depth.copy()
			overlapArea_previous = genOverlapAreaOnCurrentView(previous_depth, goal_depth, previous_pose, goal_pose)
			overlapArea_goal = genOverlapAreaOnGoalView(previous_depth, goal_depth, previous_pose, goal_pose)
			error_overlapArea_previous = genErrorOverlapAreaOnCurrentView(previous_depth, goal_depth, previous_pose, goal_pose)
			error_overlapArea_goal = genErrorOverlapAreaOnGoalView(previous_depth, goal_depth, previous_pose, goal_pose)

			fig = plt.figure(figsize=(10, 15))
			r, c, = 3, 2
			ax = fig.add_subplot(r, c, 1)
			ax.imshow(previous_img)
			ax = fig.add_subplot(r, c, 2)
			ax.imshow(goal_img)
			ax = fig.add_subplot(r, c, 3)
			ax.imshow(overlapArea_previous)
			ax = fig.add_subplot(r, c, 4)
			ax.imshow(overlapArea_goal)
			ax = fig.add_subplot(r, c, 5)
			ax.imshow(error_overlapArea_previous)
			ax = fig.add_subplot(r, c, 6)
			ax.imshow(error_overlapArea_goal)
			img_name = 'step_{}.jpg'.format(i)
			plt.savefig('{}/{}'.format(point_folder, img_name), bbox_inches='tight')
			plt.close()
			'''

		## decide if this run is successful or not
		flag_correct, dist, theta_change = similar_location_under_certainThreshold(goal_pose, list_result_poses[count_steps])
		print('dist = {:.4f}, theta = {:.4f}'.format(dist, theta_change))
		#print('start_pose = {}, final_pose = {}, goal_pose = {}'.format(start_pose, list_result_poses[-1], goal_pose))
		if flag_correct:
			count_correct += 1
			list_correct_img_names.append(right_img_name[10:])

		if flag_correct: 
			str_succ = 'Success'
		else:
			str_succ = 'Failure'

		## plot the pose graph
		#'''
		img_name = 'goTo_{}.jpg'.format(right_img_name[10:])
		print('img_name = {}'.format(img_name))

		## plot the poses
		free2 = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt/free.png'.format(scene_name), 1)
		rows, cols, _ = free2.shape
		plt.imshow(free2)

		for m in range(len(list_result_poses)):
			pose = list_result_poses[m]
			x, y = path_finder.point_to_pixel((pose[0], pose[1]))
			theta = pose[2]
			plt.arrow(x, y, cos(theta), sin(theta), color='y', \
				overhang=1, head_width=0.1, head_length=0.15, width=0.001)
		## draw goal pose
		x, y = path_finder.point_to_pixel((goal_pose[0], goal_pose[1]))
		theta = goal_pose[2]
		plt.arrow(x, y, cos(theta), sin(theta), color='r', \
				overhang=1, head_width=0.1, head_length=0.15, width=0.001)

		plt.axis([0, cols, 0, rows])
		plt.xticks([])
		plt.yticks([])
		plt.title('{}, start point_{}, goal viewpoint {}, {}\n dist = {:.4f} meter, theta = {:.4f} degree\n'.format(scene_name, point_idx, right_img_name[10:], str_succ, dist, theta_change))
		plt.savefig('{}/{}'.format(point_folder, img_name), bbox_inches='tight', dpi=(400))
		plt.close()
		#'''

		## compute stats
		current_test_dict = {}
		current_test_dict['img_name'] = right_img_name
		current_test_dict['success_flag'] = str_succ
		current_test_dict['dist'] = dist
		current_test_dict['theta'] = theta_change
		current_test_dict['steps'] = count_steps

		list_whole_stat.append(current_test_dict)

	np.save('{}/runs_statistics.npy'.format(point_folder), list_whole_stat)

	success_rate = 1.0 * count_correct / (len(point_pose_npy_file)-1)
	print('count_correct/num_right_images = {}/{} = {}'.format(count_correct, len(point_pose_npy_file)-1, success_rate))

	## write correctly run target image names to file
	f = open('{}/successful_runs.txt'.format(point_folder), 'w')
	f.write('count_correct/num_right_images = {}/{} = {}\n'.format(count_correct, len(point_pose_npy_file)-1, success_rate))
	for i in range(len(list_correct_img_names)):
		f.write('{}\n'.format(list_correct_img_names[i]))
	f.close()
	print('writing correct run image names to txt ...')


'''
if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--scene_idx', type=int, default=0)
	parser.add_argument('--category', type=str, default='Train')
	args = parser.parse_args()
	main(args.scene_idx)
'''