import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import math
from math import sin, cos
from util_visual_servoing import get_train_test_scenes, get_mapper, get_mapper_scene2points, create_folder, get_mapper_dist_theta_heading, get_pose_from_name, expand_target_object_img_to_desired_size, detect_correspondences
import glob
import sys
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/rrt')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/keypointNet')
import rrt
import utils_keypointNet as kpNet
import random
from math import cos, sin, pi
from util import action2pose, func_pose2posAndorn, similar_location_under_certainThreshold, plus_theta_fn, minus_theta_fn

'''
target_obj_name = 'sofa1'
scene_idx = 1
point_idx = 1
mask = [25, 45, 175, 188]
'''

'''
target_obj_name = 'sofa2'
scene_idx = 0
point_idx = 0
mask = [119, 106, 179, 172]
'''

'''
target_obj_name = 'rubbish_can'
scene_idx = 0
point_idx = 0
mask = [33, 106, 126, 254]
'''

'''
target_obj_name = 'picture2'
scene_idx = 0
point_idx = 0
mask = [52, 55, 112, 105]
'''

'''
target_obj_name = 'picture'
scene_idx = 2
point_idx = 4
mask = [107, 57, 135, 106]
'''

#'''
target_obj_name = 'door'
scene_idx = 2
point_idx = 4
mask = [154, 62, 193, 136]
#'''

mapper_scene2z = get_mapper()
mapper_scene2points = get_mapper_scene2points()

## create test folder
base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/test_target_objects'
approach_folder = '{}/learnedCorrespondence_gtDepth_Vx_with_mask'.format(base_folder)
#approach_folder = '{}/SIFT_gtDepth_Vx'.format(base_folder)
create_folder(approach_folder)

Train_Scenes, Test_Scenes = get_train_test_scenes()
scene_name = Test_Scenes[scene_idx]

point_folder = '{}/target_{}_scene_{}_point_{}'.format(approach_folder, target_obj_name, scene_name, point_idx)
create_folder(point_folder)

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
	return obs_rgb, obs_depth

def build_L_matrix(kp):
	lambda_focal = 128.0
	num_points = kp.shape[1]
	u0 = lambda_focal
	v0 = lambda_focal
	L = np.empty((2*num_points, 3))
	for i in range(num_points):
		v, u, Z = kp[:, i]
		u = u - u0
		v = v - v0
		Z = Z+0.0001
		L[2*i, :]   = np.array([-lambda_focal/Z, u/Z, -lambda_focal-u*u/lambda_focal])
		L[2*i+1, :] = np.array([0, v/Z, -u*v/lambda_focal])
	return L

## read in start img and start pose
sampled_image_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_test/{}/point_{}'.format(scene_name, point_idx)
start_img = cv2.imread('{}/left_img.png'.format(sampled_image_folder))[:, :, ::-1]
start_pose = mapper_scene2points[scene_name][point_idx]

count_correct = 0
list_correct_img_names = []
list_whole_stat = []
	
flag_correct = False
seq_len = 50
count_steps = 0
list_actions = []
flag_broken = False

current_img = start_img.copy()
goal_img = cv2.imread('{}/{}_mask.png'.format('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/target_objects', target_obj_name), 1)[:,:,::-1]
#goal_img = expand_target_object_img_to_desired_size(goal_img)


current_pose = [start_pose[0], start_pose[1], start_pose[2]]

## create folder for matches between two images
'''
matches_file_addr = '{}/pair_{}'.format('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/test_vs_jacobian', pair_idx)
flag_exist = os.path.isdir(matches_file_addr)
if not flag_exist:
    print('{} folder does not exist, so create one.'.format(matches_file_addr))
    os.makedirs(matches_file_addr)
'''

list_obs = [start_img for n in range(seq_len*2)]
list_result_poses = [start_pose]
num_matches = 0
while count_steps < seq_len:
	current_img, current_depth = get_obs(current_pose)
	#list_obs[count_steps] = current_img.copy()
	try:
		kp1, kp2 = kpNet.detect_learned_correspondences(current_img, goal_img)
		#kp1, kp2 = detect_correspondences(current_img, goal_img)
	except:
		print('run into error')
		break

	## remove kps in the mask
	good = []
	x_left, y_left, x_right, y_right = mask
	for i in range(kp1.shape[1]):
		v_prime = kp2[0, i]
		u_prime = kp2[1, i]
		if u_prime <= x_right and u_prime >= x_left and v_prime >= y_left and v_prime <= y_right:
			good.append(i)
	kp1 = kp1[:, good]
	kp2 = kp2[:, good]

	num_matches = kp1.shape[1]

	kp1_Z = np.empty((3, num_matches))
	kp1_Z[:2, :] = kp1
	for i in range(num_matches):
		kp1_Z[2, i] = current_depth[int(kp1_Z[0, i]), int(kp1_Z[1, i])]
	## build L matrix
	L = build_L_matrix(kp1_Z)
	## updating the projection errors
	e = kp1[::-1,:].flatten('F') - kp2[::-1,:].flatten('F')
	vc = -0.5*LA.pinv(L).dot(e)

	vx, vz, omegay= 0.5 * vc
	omegay = -omegay
	#print('omegay = {}'.format(omegay))
	#print('vz = {}'.format(vz))
	x, y, theta = current_pose
	vx_theta = minus_theta_fn(pi/2, theta)
	x = x + vz * cos(theta) + vx * cos(vx_theta)
	y = y + vz * sin(theta) + vx * sin(vx_theta)
	theta = theta + omegay
	current_pose = [x, y, theta]

	displacement = np.sum(e**2) / num_matches
	if  displacement < 25:
		print('break')
		break

	## check if the new step is on free space or not
	if not path_finder.pc.free_point(current_pose[0], current_pose[1]):
		print('run into nonfree space')
		flag_broken = True
		break

	count_steps += 1
	list_result_poses.append(current_pose)
	## sample current_img again to save in list_obs
	current_img, current_depth = get_obs(current_pose)
	list_obs[count_steps] = current_img.copy()

fig = plt.figure(figsize=(11, 10)) #cols, rows
r, c = 7, 10

## start image and goal image
ax = fig.add_subplot(r, c, 1)
ax.imshow(start_img, shape=(256, 256))
ax.title.set_text('start view')
ax.axis('off')
ax = fig.add_subplot(r, c, 2)
ax.imshow(goal_img, shape=(256, 256))
ax.title.set_text('goal view')
ax.axis('off')

for j in range(count_steps+1):
	right_img = list_obs[j]
	ax = fig.add_subplot(r, c, j+11)
	ax.imshow(right_img, shape=(256, 256))
	ax.title.set_text('step {}'.format(j))
	ax.axis('off')
if flag_correct: 
	temp_str = 'Success'
else:
	temp_str = 'Failure'
fig.suptitle('{}, start point_{}, targetObj {}'.format(scene_name, point_idx, target_obj_name))
fig.subplots_adjust(top=0.90)
fig.savefig('{}/goTo_{}.jpg'.format(point_folder, target_obj_name), bbox_inches='tight', dpi=200)
plt.close(fig)

final_img = list_obs[count_steps]
kp1, kp2 = kpNet.detect_learned_correspondences(final_img, goal_img)
## remove kps in the mask
good = []
x_left, y_left, x_right, y_right = mask
for i in range(kp1.shape[1]):
	v_prime = kp2[0, i]
	u_prime = kp2[1, i]
	if u_prime <= x_right and u_prime >= x_left and v_prime >= y_left and v_prime <= y_right:
		good.append(i)
kp1 = kp1[:, good]
kp2 = kp2[:, good]
img_combined = np.concatenate((final_img, goal_img), axis=1)
plt.imshow(img_combined)
plt.plot(kp1[1, :], kp1[0, :], 'ro', alpha=0.2)
plt.plot(kp2[1, :]+256, kp2[0, :], 'ro', alpha=0.2)
for i in range(num_matches):
	plt.plot([kp1[1, :], kp2[1, :]+256], 
		[kp1[0, :], kp2[0, :]], 'ro-', alpha=0.2)
#plt.show()
plt.title('{}, start point_{}, targetObj {}'.format(scene_name, point_idx, target_obj_name))
plt.savefig('{}/goTo_{}_matches.jpg'.format(point_folder, target_obj_name), bbox_inches='tight', dpi=200)
plt.close()


'''
if __name__ == "__main__":	
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--category_idx', type=int, default=0)
	parser.add_argument('--scene_idx', type=int, default=0)
	parser.add_argument('--point_idx', type=int, default=0)
	args = parser.parse_args()
	main(args.category_idx, args.scene_idx, args.point_idx)
'''