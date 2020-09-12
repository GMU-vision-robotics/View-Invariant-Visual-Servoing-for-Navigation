import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import math
from math import sin, cos
from util_visual_servoing import get_train_test_scenes, get_mapper, detect_correspondences

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
from util import func_pose2posAndorn

Train_Scenes, Test_Scenes = get_train_test_scenes()
scene_idx = 6
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

def get_obs(current_pose):
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

def compute_new_pose(current_pose, dist=0.1):
	x0, y0, theta0 = current_pose
	x1 = x0 + dist * math.cos(theta0)
	y1 = y0 + dist * math.sin(theta0)
	return [x1, y1, theta0]

testing_image_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/testing_image_pairs'
pair_idx = 1

current_img = cv2.imread('{}/pair_{}_left.png'.format(testing_image_folder, pair_idx))[:,:,::-1]
goal_img = cv2.imread('{}/pair_{}_right.png'.format(testing_image_folder, pair_idx))[:,:,::-1]

presampled_poses = np.load('{}/traj_poses_{}.npy'.format(testing_image_folder, pair_idx))
current_pose = presampled_poses[0]
goal_pose = compute_new_pose(current_pose)
presampled_poses = []
presampled_poses.append(current_pose)
presampled_poses.append(goal_pose)

base_file_addr = '{}'.format('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/try_depth_estimation')

current_img, current_depth = get_obs(current_pose)
goal_img, goal_depth = get_obs(goal_pose)

pair_idx = 8
cv2.imwrite('{}/pair_{}_left.png'.format(testing_image_folder, pair_idx), current_img[:,:,::-1])
cv2.imwrite('{}/pair_{}_right.png'.format(testing_image_folder, pair_idx), goal_img[:,:,::-1])
np.save('{}/pair_{}_left_depth.npy'.format(testing_image_folder, pair_idx), current_depth)
np.save('{}/pair_{}_right_depth.npy'.format(testing_image_folder, pair_idx), goal_depth)
np.save('{}/traj_poses_{}.npy'.format(testing_image_folder, pair_idx), presampled_poses)

'''
kp1, kp2 = detect_correspondences(current_img, goal_img)
num_matches = kp1.shape[1]

img_combined = np.concatenate((current_img, goal_img), axis=1)
plt.imshow(img_combined)
plt.plot(kp1[1, :], kp1[0, :], 'ro')
plt.plot(kp2[1, :]+256, kp2[0, :], 'ro')
for i in range(num_matches):
	plt.plot([kp1[1, :], kp2[1, :]+256], 
		[kp1[0, :], kp2[0, :]], 'ro-')
#plt.show()
plt.savefig('{}/step_{}.jpg'.format(matches_file_addr, count_steps), bbox_inches='tight')
plt.close()

kp1_Z = np.empty((3, num_matches))
kp1_Z[:2, :] = kp1
#kp2_Z = np.empty((3, num_matches))
#kp2_Z[:2, :] = kp2
for i in range(num_matches):
	kp1_Z[2, i] = current_depth[int(kp1_Z[0, i]), int(kp1_Z[1, i])]
## build L matrix
L = build_L_matrix(kp1_Z)
## updating the projection errors
e = kp1[::-1,:].flatten('F') - kp2[::-1,:].flatten('F')
vc = -0.5*LA.pinv(L).dot(e)
'''


