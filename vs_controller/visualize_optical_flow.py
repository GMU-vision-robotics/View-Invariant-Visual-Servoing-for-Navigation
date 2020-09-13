import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import cv2
import sys
from collections import deque
from math import sin, cos, sqrt, pi, atan2
import random
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop')
from util_visual_servoing import get_train_test_scenes, get_mapper, get_mapper_scene2points, create_folder, sample_gt_dense_correspondences
from util import plus_theta_fn, minus_theta_fn
from util_vscontroller import genGtDenseCorrespondenseFlowMap, normalize_opticalFlow, normalize_depth


scene_idx = 2


mapper_scene2z = get_mapper()
mapper_scene2points = get_mapper_scene2points()
Train_Scenes, Test_Scenes = get_train_test_scenes()

scene_name = Test_Scenes[scene_idx]
num_startPoints = len(mapper_scene2points[scene_name])

base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_test'

save_base_folder = '/home/reza/Datasets/GibsonEnv/my_code/vs_controller/optical_flow'
create_folder(save_base_folder)

#for point_idx in range(0, 1):
for point_idx in range(2, 3):
	print('point_idx = {}'.format(point_idx))

	save_point_folder = '{}/{}_point_{}'.format(save_base_folder, scene_name, point_idx)
	create_folder(save_point_folder)

	## read in start img and start pose
	point_image_folder = '{}/{}/point_{}'.format(base_folder, scene_name, point_idx)
	point_pose_npy_file = np.load('{}/{}/point_{}_poses.npy'.format(base_folder, scene_name, point_idx))

	start_img = cv2.imread('{}/{}.png'.format(point_image_folder, point_pose_npy_file[0]['img_name']))[:, :, ::-1]
	start_depth = np.load('{}/{}_depth.npy'.format(point_image_folder, point_pose_npy_file[0]['img_name']))
	start_pose = point_pose_npy_file[0]['pose']

	for right_img_idx in range(1, len(point_pose_npy_file)):
	#for right_img_idx in range(1, 2):
		print('right_img_idx = {}'.format(right_img_idx))

		right_img_name = point_pose_npy_file[right_img_idx]['img_name']
		goal_img = cv2.imread('{}/{}.png'.format(point_image_folder, right_img_name), 1)[:,:,::-1]
		goal_depth = np.load('{}/{}_depth.npy'.format(point_image_folder, right_img_name))
		goal_pose = point_pose_npy_file[right_img_idx]['pose']

		optical_flow = genGtDenseCorrespondenseFlowMap(start_depth, goal_depth, start_pose, goal_pose)[:,:,:2]
		## minmax normalization to visualize optical_flow, as it contains negative float numbers
		
		'''
		normalized_depth = normalize_depth(start_depth)
		new_optical_flow = optical_flow * normalized_depth
		#assert 1==2
		normalized_optical_flow = np.zeros((256, 256, 3), dtype=np.float32)
		normalized_optical_flow[:, :, :2] = normalize_opticalFlow(new_optical_flow)
		'''

		'''
		fig = plt.figure(figsize=(25, 5))
		r, c, = 1, 4
		ax = fig.add_subplot(r, c, 1)
		ax.imshow(start_img)
		ax.axis('off')
		ax = fig.add_subplot(r, c, 2)
		ax.imshow(goal_img)
		ax.axis('off')
		ax = fig.add_subplot(r, c, 3)
		im1 = ax.imshow(optical_flow[:, :, 0], vmin=-100, vmax=100, cmap='viridis')
		ax.axis('off')
		fig.colorbar(im1, fraction=0.046, pad=0.04)
		ax = fig.add_subplot(r, c, 4)
		im = ax.imshow(optical_flow[:, :, 1], vmin=-100, vmax=100, cmap='viridis')
		ax.axis('off')
		fig.colorbar(im, fraction=0.046, pad=0.04)
		fig.savefig('{}/goTo_{}.jpg'.format(save_point_folder, right_img_name), bbox_inches='tight')
		plt.close(fig)

		'''

		cv2.imwrite('{}/goTo_{}_start.jpg'.format(save_point_folder, right_img_name), start_img[:,:,::-1])
		cv2.imwrite('{}/goTo_{}_goal.jpg'.format(save_point_folder, right_img_name), goal_img[:,:,::-1])
		plt.imshow(optical_flow[:, :, 0], vmin=-100, vmax=100, cmap='viridis')
		plt.axis('off')
		plt.savefig('{}/goTo_{}_left.jpg'.format(save_point_folder, right_img_name), bbox_inches='tight', pad_inches=0)
		plt.close()
		plt.imshow(optical_flow[:, :, 1], vmin=-100, vmax=100, cmap='viridis')
		plt.axis('off')
		plt.savefig('{}/goTo_{}_right.jpg'.format(save_point_folder, right_img_name), bbox_inches='tight', pad_inches=0)
		plt.close()
