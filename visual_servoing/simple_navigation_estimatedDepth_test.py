import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import math
from math import sin, cos, pi
from util_visual_servoing import get_train_test_scenes, get_mapper, detect_correspondences_and_descriptors, detect_correspondences_for_fixed_kps, estimate_depth, compute_new_pose

import sys
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/rrt')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop')
import rrt
import random
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

'''
def compute_new_pose(current_pose, dist=0.1):
	x0, y0, theta0 = current_pose
	x1 = x0 + dist * math.cos(theta0)
	y1 = y0 + dist * math.sin(theta0)
	return [x1, y1, theta0]
'''

def get_obs(current_pose):
	#print('current_pose = {}'.format(current_pose))
	pos, orn = func_pose2posAndorn(current_pose, mapper_scene2z[scene_name])
	env.robot.reset_new_pose(pos, orn)
	obs, _, _, _ = env.step(4)
	obs_rgb = obs['rgb_filled']
	obs_depth = obs['depth']
	return obs_rgb.copy(), obs_depth.copy()

def build_L_matrix(kp):
	lambda_focal = 128.0
	num_points = kp.shape[1]
	u0 = lambda_focal
	v0 = lambda_focal
	L = np.empty((2*num_points, 2))
	for i in range(num_points):
		v, u, Z = kp[:, i]
		u = u - u0
		v = v - v0
		Z = Z+0.0001
		L[2*i, :]   = np.array([u/Z, -lambda_focal-u*u/lambda_focal])
		L[2*i+1, :] = np.array([v/Z, -u*v/lambda_focal])
	return L

testing_image_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/testing_image_pairs'
pair_idx = 7

#current_img = cv2.imread('{}/pair_{}_left.png'.format(testing_image_folder, pair_idx))[:,:,::-1]
#goal_img = cv2.imread('{}/pair_{}_right.png'.format(testing_image_folder, pair_idx))[:,:,::-1]

presampled_poses = np.load('{}/traj_poses_{}.npy'.format(testing_image_folder, pair_idx))
current_pose = presampled_poses[0]
current_img, _ = get_obs(current_pose)

goal_pose = presampled_poses[-1]
goal_img, _ = get_obs(goal_pose)

seq_len = 20
count_steps = 0
list_result_poses = [current_pose]

flag_broken = False
list_obs = [current_img.copy() for n in range(seq_len*2)]

while count_steps < seq_len:
	print('************************  count_steps = {}  *************************'.format(count_steps))
	current_img, current_depth = get_obs(current_pose)

	kp1, kp2, des1, _ = detect_correspondences_and_descriptors(current_img, goal_img)
	
	if count_steps == 0:
		new_pose = compute_new_pose(current_pose)
		new_img, _ = get_obs(new_pose)
	else:
		new_img = list_obs[count_steps-1]
		new_pose = list_result_poses[count_steps-1]
	print('kp1.shape = {}'.format(kp1.shape))
	kp3, good = detect_correspondences_for_fixed_kps(des1, new_img)
	kp1 = kp1[:, good]
	kp2 = kp2[:, good]
	print('kp1.shape = {}'.format(kp1.shape))
	assert kp3.shape == kp1.shape
	#print('current_pose = {}, new_pose = {}'.format(current_pose, new_pose))
	print('kp1 = {}'.format(kp1))
	print('kp3 = {}'.format(kp3))
	print('current_pose = {}, new_pose = {}'.format(current_pose, new_pose))
	kp1_estimated_Zs = estimate_depth(kp1, kp3, current_pose, new_pose)
	#print('estimated_Z = {}'.format(kp1_estimated_Zs))
	
	num_matches = kp1.shape[1]
	for i in range(num_matches):
		v, u = kp1[:, i]
		print('gt_depth = {}, estimated_depth = {:.5f}'.format(current_depth[int(v), int(u)], kp1_estimated_Zs[i]))
	
	img_combined = np.concatenate((current_img, new_img), axis=1)
	plt.imshow(img_combined)
	plt.plot(kp1[1, :], kp1[0, :], 'ro')
	plt.plot(kp3[1, :]+256, kp3[0, :], 'ro')
	for i in range(num_matches):
		plt.plot([kp1[1, :], kp3[1, :]+256], 
			[kp1[0, :], kp3[0, :]], 'ro-')
	file_addr = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/test_vs_estimated_depth'
	plt.savefig('{}/pair_{}_step_{}_new_img.jpg'.format(file_addr, pair_idx, count_steps), bbox_inches='tight')
	plt.close()
	img_combined = np.concatenate((current_img, goal_img), axis=1)
	plt.imshow(img_combined)
	plt.plot(kp1[1, :], kp1[0, :], 'ro')
	plt.plot(kp2[1, :]+256, kp2[0, :], 'ro')
	for i in range(num_matches):
		plt.plot([kp1[1, :], kp2[1, :]+256], 
			[kp1[0, :], kp2[0, :]], 'ro-')
	#plt.show()
	file_addr = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/test_vs_estimated_depth'
	plt.savefig('{}/pair_{}_step_{}_goal_img.jpg'.format(file_addr, pair_idx, count_steps), bbox_inches='tight')
	plt.close()
	#assert 1==2
	'''
	print('-----------------------------------------------------------------------------------------------')
	kp1_Z = np.empty((3, num_matches))
	kp1_Z[:2, :] = kp1
	#kp2_Z = np.empty((3, num_matches))
	#kp2_Z[:2, :] = kp2
	for i in range(num_matches):
		v, u = kp1[:, i]
		kp1_Z[2, i] = current_depth[int(v), int(u)]
	## build L matrix
	L = build_L_matrix(kp1_Z)
	## updating the projection errors
	e = kp1[::-1,:].flatten('F') - kp2[::-1,:].flatten('F')
	vc = -0.5*LA.pinv(L).dot(e)
	vz, omegay= 0.5 * vc
	omegay = -omegay
	print('gt omegay = {}'.format(omegay))
	print('gt vz = {}'.format(vz))

	for i in range(num_matches):
		current_kp1 = np.expand_dims(kp1_Z[:, i], axis=1) 
		## build L matrix
		L = build_L_matrix(current_kp1)
		## updating the projection errors
		e = kp1[::-1, i].flatten('F') - kp2[::-1, i].flatten('F')
		vc = -0.5*LA.pinv(L).dot(e)
		vz, omegay= 0.5 * vc
		omegay = -omegay
		print('using match {}, omegay = {} vz = {}'.format(i, omegay, vz))
	'''
	print('======================================================================================')
	kp1_Z = np.empty((3, num_matches))
	kp1_Z[:2, :] = kp1
	#kp2_Z = np.empty((3, num_matches))
	#kp2_Z[:2, :] = kp2
	for i in range(num_matches):
		kp1_Z[2, i] = kp1_estimated_Zs[i]
	## build L matrix
	L = build_L_matrix(kp1_Z)
	## updating the projection errors
	e = kp1[::-1,:].flatten('F') - kp2[::-1,:].flatten('F')
	vc = -0.5*LA.pinv(L).dot(e)

	vz, omegay= vc
	omegay = -omegay
	print('omegay = {}'.format(omegay))
	print('vz = {}'.format(vz))
	'''
	for i in range(num_matches):
		current_kp1 = np.expand_dims(kp1_Z[:, i], axis=1) 
		## build L matrix
		L = build_L_matrix(current_kp1)
		## updating the projection errors
		e = kp1[::-1, i].flatten('F') - kp2[::-1, i].flatten('F')
		vc = -0.5*LA.pinv(L).dot(e)
		vz, omegay= 0.5 * vc
		omegay = -omegay
		print('using match {}, omegay = {} vz = {}'.format(i, omegay, vz))
	'''
	print('======================================================================================')
	#if count_steps == 1:
	#	assert 1==2
	#assert 1==2
	x, y, theta = current_pose
	x = x + vz * cos(theta)
	y = y + vz * sin(theta)
	theta = theta + omegay
	current_pose = x, y, theta

	## check if we should stop or not
	if np.sum(e**2) / num_matches < 25:
		print('break')
		break

	## check if the new step is on free space or not
	if not path_finder.pc.free_point(current_pose[0], current_pose[1]):
		flag_broken = True
		break
	
	count_steps += 1
	list_result_poses.append(current_pose)
	current_img, _ = get_obs(current_pose)
	list_obs[count_steps] = current_img.copy()

## plot the pose graph
file_addr = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/test_vs_estimated_depth'
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

#np.save('{}/actions_pair_{}.npy'.format(file_addr, pair_idx), np.array(list_actions))