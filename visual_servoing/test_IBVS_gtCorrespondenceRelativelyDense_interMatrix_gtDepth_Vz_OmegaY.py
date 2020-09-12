import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import math
from math import sin, cos
from util_visual_servoing import get_train_test_scenes, get_mapper, detect_correspondences, get_mapper_scene2points, create_folder, get_mapper_dist_theta_heading, get_pose_from_name, sample_gt_correspondences_with_large_displacement, build_L_matrix, compute_velocity_through_correspondences_and_depth, goToPose_one_step, sample_gt_correspondences_relativelyDense
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
perception_rep = 'gtSparseRelativelyDense' # 'gtDense', 'learnedSparse', 'sift'
depth_method = 'gt' # 'estimated', 'void'

def main(scene_idx=0):

	#scene_idx = 0

	mapper_scene2z = get_mapper()
	mapper_scene2points = get_mapper_scene2points()
	Train_Scenes, Test_Scenes = get_train_test_scenes()
	scene_name = Test_Scenes[scene_idx]
	num_startPoints = len(mapper_scene2points[scene_name])
	num_steps = 50

	## create test folder
	test_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/test_IBVS'
	#approach_folder = '{}/gtCorrespondence_interMatrix_gtDepth_Vz_OmegaY'.format(test_folder)
	#create_folder(approach_folder)

	#scene_folder = '{}/{}'.format(approach_folder, scene_name)
	#create_folder(scene_folder)

	f = open('{}/{}_{}.txt'.format(test_folder, perception_rep, depth_method), 'a')
	f.write('scene_name = {}\n'.format(scene_name))
	list_count_correct = []
	list_count_runs = []

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
	for point_idx in range(0, num_startPoints):
	#for point_idx in range(0, 1):
		print('point_idx = {}'.format(point_idx))

		#point_folder = '{}/point_{}'.format(scene_folder, point_idx)
		#create_folder(point_folder)

		## read in start img and start pose
		point_image_folder = '{}/{}/point_{}'.format(base_folder, scene_name, point_idx)
		point_pose_npy_file = np.load('{}/{}/point_{}_poses.npy'.format(base_folder, scene_name, point_idx))

		start_img = cv2.imread('{}/{}.png'.format(point_image_folder, point_pose_npy_file[0]['img_name']))[:, :, ::-1]
		start_pose = point_pose_npy_file[0]['pose']

		## index 0 is the left image, so right_img_idx starts from index 1
		count_correct = 0
		list_correct_img_names = []
		list_whole_stat = []

		for right_img_idx in range(1, len(point_pose_npy_file)):
		#for right_img_idx in range(10, 11):
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
			while count_steps < num_steps:
				current_img, current_depth = get_obs(current_pose)
				try:
					kp1, kp2 = sample_gt_correspondences_relativelyDense(current_depth, goal_depth, current_pose, goal_pose)
					if count_steps == 0:
						start_depth = current_depth.copy()
				except:
					print('run into error')
					break
				num_matches = kp1.shape[1]
				print('num_matches = {}'.format(num_matches))

				vx, vz, omegay, flag_stop = compute_velocity_through_correspondences_and_depth(kp1, kp2, current_depth)

				previous_pose = current_pose
				current_pose, _, _, flag_stop_goToPose = goToPose_one_step(current_pose, vx, vz, omegay)

				## check if there is collision during the action
				left_pixel = path_finder.point_to_pixel((previous_pose[0], previous_pose[1]))
				right_pixel = path_finder.point_to_pixel((current_pose[0], current_pose[1]))
				# rrt.line_check returns True when there is no obstacle
				if not rrt.line_check(left_pixel, right_pixel, free):
					flag_broken = True
					#print('run into an obstacle ...')
					break

				## check if we should stop or not
				if flag_stop or flag_stop_goToPose:
					#print('flag_stop = {}, flag_stop_goToPose = {}'.format(flag_stop, flag_stop_goToPose))
					#print('break')
					break

				count_steps += 1
				list_result_poses.append(current_pose)
				## sample current_img again to save in list_obs
				current_img, current_depth = get_obs(current_pose)
			#assert 1==2
			## decide if this run is successful or not
			flag_correct, dist, theta_change = similar_location_under_certainThreshold(goal_pose, list_result_poses[count_steps])
			#print('dist = {}, theta = {}'.format(dist, theta_change))
			#print('start_pose = {}, final_pose = {}, goal_pose = {}'.format(start_pose, list_result_poses[-1], goal_pose))
			if flag_correct:
				count_correct += 1
				list_correct_img_names.append(right_img_name[10:])

			if flag_correct: 
				str_succ = 'Success'
				print('str_succ = {}'.format(str_succ))
			else:
				str_succ = 'Failure'
				print('str_succ = {}'.format(str_succ))

			## ===================================================================================================================
			## plot the pose graph
			'''
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

			## ======================================================================================================================
			## save stats
			current_test_dict = {}
			current_test_dict['img_name'] = right_img_name
			current_test_dict['success_flag'] = flag_correct
			current_test_dict['dist'] = dist
			current_test_dict['theta'] = theta_change
			current_test_dict['steps'] = count_steps
			current_test_dict['collision'] = flag_broken

			list_whole_stat.append(current_test_dict)
			'''

		'''
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

		f.write('point {} : {}/{}\n'.format(point_idx, count_correct, len(point_pose_npy_file)-1))
		f.flush()
		list_count_correct.append(count_correct)
		list_count_runs.append(len(point_pose_npy_file)-1)

	f.write('In total : {}/{}\n'.format(sum(list_count_correct), sum(list_count_runs)))
	f.write('-------------------------------------------------------------------------------------\n')	

#'''
if __name__ == "__main__":	
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--scene_idx', type=int, default=0)
	args = parser.parse_args()
	main(args.scene_idx)
#'''