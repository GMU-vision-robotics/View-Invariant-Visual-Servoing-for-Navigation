import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import cv2
import sys
from collections import deque
from math import sin, cos, sqrt, pi, atan2, floor
import random
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop')
from util_visual_servoing import get_train_test_scenes, get_mapper, get_mapper_scene2points, create_folder, sample_gt_dense_correspondences
from util import plus_theta_fn, minus_theta_fn
from util_vscontroller import genGtDenseCorrespondenseFlowMap, genGtDenseCorrespondenseFlowMapOnObjects

def normalize_opticalFlow(optical_flow):
	assert optical_flow.shape[2] == 2
	for c in range(2):
		channel = optical_flow[:, :, c]
		optical_flow[:, :, c] = (channel - (-256)) / 512
	return optical_flow

def normalize_depth(depth, confident_dist=5.0):
	assert depth.shape[2] == 1
	depth = depth / confident_dist
	depth[depth > 1] = 1.0
	return depth


scene_idx = 1


mapper_scene2z = get_mapper()
mapper_scene2points = get_mapper_scene2points()
Train_Scenes, Test_Scenes = get_train_test_scenes()

scene_name = Test_Scenes[scene_idx]
num_startPoints = len(mapper_scene2points[scene_name])

base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_test'

save_base_folder = '/home/reza/Datasets/GibsonEnv/my_code/vs_controller/optical_flow'
create_folder(save_base_folder)

for point_idx in range(0, 1):
	print('point_idx = {}'.format(point_idx))

	save_point_folder = '{}/{}_point_{}'.format(save_base_folder, scene_name, point_idx)
	create_folder(save_point_folder)

	## read in start img and start pose
	point_image_folder = '{}/{}/point_{}'.format(base_folder, scene_name, point_idx)
	point_pose_npy_file = np.load('{}/{}/point_{}_poses.npy'.format(base_folder, scene_name, point_idx))
	point_detection_npy_file = np.load('{}/{}/point_{}_detections.npy'.format(base_folder, scene_name, point_idx))

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

		# read detections
		goal_bbox = np.floor(point_detection_npy_file[right_img_idx]['bbox']).astype(int)
		
		if goal_bbox.shape[0] > 0:
			goal_bbox = goal_bbox[0]

			optical_flow, flag_having_correspondence = genGtDenseCorrespondenseFlowMapOnObjects(start_depth, goal_depth, start_pose, goal_pose, goal_bbox)
			optical_flow = optical_flow[:, :, :2]
			normalize_optical_flow = np.zeros((256, 256, 3), dtype=np.float32)
			normalize_optical_flow[:, :, :2] = normalize_opticalFlow(optical_flow)
			
			# visualize the object
			x1, y1, x2, y2 = goal_bbox
			center_bbox_x, center_bbox_y = floor((x1+x2)/2), floor((y1+y2)/2)
			trans_x, trans_y = 128 - center_bbox_x, 128 - center_bbox_y
			object_img = np.zeros((256, 256, 3), dtype=np.uint8)
			object_img[y1+trans_y:y2+trans_y, x1+trans_x:x2+trans_x, :] = goal_img[y1:y2, x1:x2, :]


			fig = plt.figure(figsize=(10, 5))
			r, c, = 1, 4
			ax = fig.add_subplot(r, c, 1)
			ax.imshow(start_img)
			ax = fig.add_subplot(r, c, 2)
			ax.imshow(goal_img)
			ax = fig.add_subplot(r, c, 3)
			ax.imshow(object_img)
			ax = fig.add_subplot(r, c, 4)
			ax.imshow(normalize_optical_flow)




			plt.show()
			#assert 1==2
			#fig.savefig('{}/goTo_{}.jpg'.format(save_point_folder, right_img_name), bbox_inches='tight')
			#plt.close(fig)
