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
from util_vscontroller import genOverlapAreaOnCurrentView, gt_goToPose, genOverlapAreaOnGoalView, 


scene_idx = 1

mapper_scene2z = get_mapper()
mapper_scene2points = get_mapper_scene2points()
Train_Scenes, Test_Scenes = get_train_test_scenes()

scene_name = Test_Scenes[scene_idx]
num_startPoints = len(mapper_scene2points[scene_name])

base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_test'

save_base_folder = '/home/reza/Datasets/GibsonEnv/my_code/vs_controller/overlapping_area'
create_folder(save_base_folder)

for point_idx in range(0, 1):
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
		print('right_img_idx = {}'.format(right_img_idx))

		right_img_name = point_pose_npy_file[right_img_idx]['img_name']
		goal_img = cv2.imread('{}/{}.png'.format(point_image_folder, right_img_name), 1)[:,:,::-1]
		goal_depth = np.load('{}/{}_depth.npy'.format(point_image_folder, right_img_name))
		goal_pose = point_pose_npy_file[right_img_idx]['pose']

		kp1, kp2 = sample_gt_dense_correspondences(start_depth, goal_depth, start_pose, goal_pose, gap=1, focal_length=128, resolution=256, start_pixel=1, depth_verification=True)

		kp1, kp2 = kp1.astype(np.int16), kp2.astype(np.int16)

		## match from goal view to current view
		start_mask = np.zeros((256, 256, 3), dtype=np.uint8)
		goal_mask = np.zeros((256, 256, 3), dtype=np.uint8)
		for i in range(256):
			for j in range(256):
				goal_mask[i, j] = [i, j, 0]

		for i in range(kp1.shape[1]):
			y1, x1 = kp1[:, i]
			y2, x2 = kp2[:, i]
			start_mask[y1, x1] = goal_mask[y2, x2]

		## build error mask for start_mask and goal_mask2
		error_start_mask = np.zeros((256, 256, 3), dtype=np.uint8)
		for i in range(256):
			for j in range(256):
				error_start_mask[i, j] = [i, j, 0]
		for i in range(kp1.shape[1]):
			y1, x1 = kp1[:, i]
			error_start_mask[y1, x1] = [0, 0, 0]

		kp2, kp1 = sample_gt_dense_correspondences(goal_depth, start_depth, goal_pose, start_pose, gap=1, focal_length=128, resolution=256, start_pixel=1, depth_verification=True)
		kp2, kp1 = kp2.astype(np.int16), kp1.astype(np.int16)

		## match from current view to goal view
		start_mask2 = np.zeros((256, 256, 3), dtype=np.uint8)
		goal_mask2 = np.zeros((256, 256, 3), dtype=np.uint8)
		for i in range(256):
			for j in range(256):
				start_mask2[i, j] = [i, j, 0]

		for i in range(kp1.shape[1]):
			y1, x1 = kp1[:, i]
			y2, x2 = kp2[:, i]
			goal_mask2[y2, x2] = start_mask2[y1, x1]
		
		## build error mask for start_mask and goal_mask2
		error_goal_mask2 = np.zeros((256, 256, 3), dtype=np.uint8)
		for i in range(256):
			for j in range(256):
				error_goal_mask2[i, j] = [i, j, 0]
		for i in range(kp2.shape[1]):
			y2, x2 = kp2[:, i]
			error_goal_mask2[y2, x2] = [0, 0, 0]

		
		fig = plt.figure(figsize=(5, 10))
		r, c, = 4, 2
		ax = fig.add_subplot(r, c, 1)
		ax.imshow(start_img)
		ax = fig.add_subplot(r, c, 2)
		ax.imshow(goal_img)
		ax = fig.add_subplot(r, c, 3)
		ax.imshow(start_mask)
		ax = fig.add_subplot(r, c, 4)
		ax.imshow(goal_mask)
		ax = fig.add_subplot(r, c, 5)
		ax.imshow(start_mask2)
		ax = fig.add_subplot(r, c, 6)
		ax.imshow(goal_mask2)
		ax = fig.add_subplot(r, c, 7)
		ax.imshow(error_start_mask)
		ax = fig.add_subplot(r, c, 8)
		ax.imshow(error_goal_mask2)
		#plt.show()
		#assert 1==2
		fig.savefig('{}/goTo_{}.jpg'.format(save_point_folder, right_img_name), bbox_inches='tight')
		plt.close(fig)
