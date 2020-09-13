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
from util_vscontroller import genGtDenseCorrespondenseFlowMap

def addGaussianNoise(dataMap, mu=0.0, sigma=10.0):
	h, w, channel = dataMap.shape
	for c in range(channel):
		noise = np.random.normal(mu, sigma, (h, w))
		dataMap[:, :, c] += noise
	dataMap[dataMap < -256.0] = -256.0
	dataMap[dataMap > 256.0] = 256.0
	return dataMap

## keep_prob indicates how many points get a correspondence.
def removeCorrespondenceRandomly(dataMap, keep_prob=1.0):
	h, w, _ = dataMap.shape
	s = np.random.uniform(0, 1, (h, w))
	low_bound = 1.0 - keep_prob
	mask = np.zeros((h, w, 1), dtype=np.float32)
	mask[s >= low_bound] = 1.0
	dataMap = dataMap * mask
	return dataMap

## keep_prob indicates how many points get a correspondence.
def removeCorrespondenceRandomly_withSmoothing(dataMap, keep_prob=1.0):
	h, w, _ = dataMap.shape
	s = np.random.uniform(0, 1, (h, w))
	low_bound = 1.0 - keep_prob
	mask = np.zeros((h, w, 1), dtype=np.float32)
	mask[s >= low_bound] = 1.0
	dataMap = dataMap * mask
	if keep_prob < 0.99:
		dataMap = cv2.blur(dataMap, (5, 5))
	return dataMap


scene_idx = 1

mapper_scene2z = get_mapper()
mapper_scene2points = get_mapper_scene2points()
Train_Scenes, Test_Scenes = get_train_test_scenes()

scene_name = Test_Scenes[scene_idx]
num_startPoints = len(mapper_scene2points[scene_name])

base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_test'

save_base_folder = '/home/reza/Datasets/GibsonEnv/my_code/vs_controller/artifical_noise'
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

	#for right_img_idx in range(1, len(point_pose_npy_file)):
	for right_img_idx in range(1, 10):
		print('right_img_idx = {}'.format(right_img_idx))

		right_img_name = point_pose_npy_file[right_img_idx]['img_name']
		goal_img = cv2.imread('{}/{}.png'.format(point_image_folder, right_img_name), 1)[:,:,::-1]
		goal_depth = np.load('{}/{}_depth.npy'.format(point_image_folder, right_img_name))
		goal_pose = point_pose_npy_file[right_img_idx]['pose']

		optical_flow = genGtDenseCorrespondenseFlowMap(start_depth, goal_depth, start_pose, goal_pose)
		## minmax normalization to visualize optical_flow, as it contains negative float numbers
		visualize_optical_flow = np.zeros((256, 256, 3), dtype=np.float32)
		for c in range(2):
			channel = optical_flow[:, :, c]
			visualize_optical_flow[:, :, c] = (channel - (-256)) / 512

		## add GaussianNoise
		gaussianNoise_optical_flow = addGaussianNoise(optical_flow, sigma=100)
		visualize_noise_optical_flow = np.zeros((256, 256, 3), dtype=np.float32)
		for c in range(2):
			channel = gaussianNoise_optical_flow[:, :, c]
			visualize_noise_optical_flow[:, :, c] = (channel - (-256)) / 512

		reduced_optical_flow = removeCorrespondenceRandomly(optical_flow, 0.5)
		visualize_reduced_optical_flow = np.zeros((256, 256, 3), dtype=np.float32)
		for c in range(2):
			channel = reduced_optical_flow[:, :, c]
			visualize_reduced_optical_flow[:, :, c] = (channel - (-256)) / 512

		smoothed_reduced_optical_flow = removeCorrespondenceRandomly_withSmoothing(optical_flow, 0.5)
		visualize_smoothed_reduced_optical_flow = np.zeros((256, 256, 3), dtype=np.float32)
		for c in range(2):
			channel = reduced_optical_flow[:, :, c]
			visualize_smoothed_reduced_optical_flow[:, :, c] = (channel - (-256)) / 512


		fig = plt.figure(figsize=(30, 5))
		r, c, = 1, 6
		ax = fig.add_subplot(r, c, 1)
		ax.imshow(start_img)
		ax = fig.add_subplot(r, c, 2)
		ax.imshow(goal_img)
		ax = fig.add_subplot(r, c, 3)
		ax.imshow(visualize_optical_flow)
		ax = fig.add_subplot(r, c, 4)
		ax.imshow(visualize_noise_optical_flow)
		ax.title.set_text('{}'.format('gaussian noise to displacement'))
		ax = fig.add_subplot(r, c, 5)
		ax.imshow(visualize_reduced_optical_flow)
		ax.title.set_text('{}'.format('missing correspondence'))
		ax = fig.add_subplot(r, c, 6)
		ax.imshow(visualize_smoothed_reduced_optical_flow)

		#plt.show()
		#assert 1==2
		fig.savefig('{}/goTo_{}.jpg'.format(save_point_folder, right_img_name), bbox_inches='tight', dpi=(400))
		plt.close(fig)
