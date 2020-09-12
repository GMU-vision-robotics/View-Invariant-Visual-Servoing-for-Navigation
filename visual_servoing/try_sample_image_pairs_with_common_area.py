import sys
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/rrt')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop')
import rrt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from math import cos, sin, pi
from util import func_pose2posAndorn, plus_theta_fn, minus_theta_fn
from util_visual_servoing import get_train_test_scenes, get_mapper

#'''
## For Gibson Env
import gym, logging
from mpi4py import MPI
from gibson.envs.husky_env import HuskyNavigateEnv
from baselines import logger
import skimage.io
from transforms3d.euler import euler2quat
#'''

random.seed(5)
np.random.seed(5)

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
free = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt/free.png'.format(scene_name), 0)

 ## GibsonEnv setup
config_file = os.path.join('/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop', 'env_yamls', '{}_navigate.yaml'.format(scene_name))
env = HuskyNavigateEnv(config=config_file, gpu_count = 1)
obs = env.reset() ## this line is important otherwise there will be an error like 'AttributeError: 'HuskyNavigateEnv' object has no attribute 'potential''


## random sample source_node_index and destin_node_index
def func_random_path_generator():
    source_node_index = random.randint(0, num_nodes)
    destin_node_index = random.randint(0, num_nodes)
    while destin_node_index == source_node_index:
        destin_node_index = random.randint(0, num_nodes)

    x0 = path_finder.nodes_x[source_node_index] 
    y0 = path_finder.nodes_y[source_node_index]
    x1 = path_finder.nodes_x[destin_node_index]
    y1 = path_finder.nodes_y[destin_node_index]

    solution, lines = path_finder.find(x0, y0, x1, y1)

    points = []
    for i in range(len(lines)):
        x, y = path_finder.pixel_to_point((lines[i][1], lines[i][0]))
        points.append((x, y))
    return points

## expand rrt generated points into executable traj including poses and actions
def func_expand_rrt_traj_points(points):
    ## threshold for forward and turning
    thresh_forward = 0.01
    upper_thresh_theta = math.pi / 6
    lower_thresh_theta = math.pi / 12

    ## compute theta for each point except the last one
    ## theta is in the range [-pi, pi]
    thetas = []
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        current_theta = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
        thetas.append(current_theta)

    assert len(thetas) == len(points) - 1

    # pose: (x, y, theta)
    poses = []
    actions = []

    previous_theta = 0
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]

        current_theta = thetas[i]
        ## so that previous_theta is same as current_theta for the first point
        if i == 0:
            previous_theta = current_theta
        ## first point is not the result of an action
        ## append an action before introduce a new pose
        if i != 0:
            ## forward: 0, left: 3, right 2
            actions.append(0)
        ## after turning, previous theta is changed into current_theta
        pose = (p1[0], p1[1], previous_theta)
        poses.append(pose)
        ## first add turning points
        ## decide turn left or turn right, Flase = left, True = Right
        bool_turn = False
        minus_cur_pre_theta = minus_theta_fn(previous_theta, current_theta)
        if minus_cur_pre_theta < 0:
            bool_turn = True
        ## need to turn more than once, since each turn is 30 degree
        while abs(minus_theta_fn(previous_theta, current_theta)) > upper_thresh_theta:
            if bool_turn:
                previous_theta = minus_theta_fn(upper_thresh_theta, previous_theta)
                actions.append(2)
            else:
                previous_theta = plus_theta_fn(upper_thresh_theta, previous_theta)
                actions.append(3)
            pose = (p1[0], p1[1], previous_theta)
            poses.append(pose)
        ## add one more turning points when change of theta > 15 degree
        if abs(minus_theta_fn(previous_theta, current_theta)) > lower_thresh_theta:
            if bool_turn:
                actions.append(2)
            else:
                actions.append(3)
            pose = (p1[0], p1[1], current_theta)
            poses.append(pose)
        ## no need to change theta any more
        previous_theta = current_theta
        ## then add forward points

        ## we don't need to add p2 to poses unless p2 is the last point in points
        if i + 1 == len(points) - 1:
            actions.append(0)
            pose = (p2[0], p2[1], current_theta)
            poses.append(pose)
    return poses, actions

def func_sampleTraj(len_traj=40):
    ## sampling poses
    while True:
        points = func_random_path_generator()
        poses, actions = func_expand_rrt_traj_points(points)
        if len(poses) > len_traj:
            break
    ## cut poses and actions into trajectory of len_traj
    len_poses = len(poses)
    ## print out forward distance between neighboring points
    start_pos_index = random.randint(0, len_poses-len_traj)
    poses = poses[start_pos_index : start_pos_index+len_traj]
    actions = actions[start_pos_index : start_pos_index+len_traj-1]
    return poses, actions



## create theta list from -45 degree to 45 degree gap 15 degree
## create distance list from 0.5 to 3.0 gap 0.5
theta_list = []
for i in range(7):
	theta = -math.pi/4 + i * math.pi/12
	theta_list.append(theta)
dist_list = [0.5*i for i in range(1, 7)]

## create pose theta list from -pi to pi gap 15 degree
pose_theta_list = []
for i in range(24):
	theta = -math.pi + i * math.pi/12
	pose_theta_list.append(theta)

## create diff_theta_list from -75 to +75 gap 15 degree
diff_theta_list = []
## -45, -30, -15, 0, 15, 30, 45
for i in range(7):
	theta = -3/12.0*math.pi + i * math.pi/12
	diff_theta_list.append(theta)

source_node_idx = random.randint(0, num_nodes-1)
x0 = path_finder.nodes_x[source_node_idx] 
y0 = path_finder.nodes_y[source_node_idx]
theta0 = pose_theta_list[random.randint(0, 23)]
left_pose = [x0, y0, theta0]

file_addr = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs'
current_pose = left_pose
pos, orn = func_pose2posAndorn(current_pose, mapper_scene2z[scene_name])
env.robot.reset_new_pose(pos, orn)
obs, _, _, _ = env.step(4)
obs_rgb = obs['rgb_filled']
cv2.imwrite('{}/left_img.png'.format(file_addr), obs_rgb[:,:,::-1])

right_pose_list = []
for i in range(len(theta_list)):
	for j in range(len(dist_list)):
		#temp_theta = plus_theta_fn(theta0, theta_list[random.randint(0, 6)])
		#temp_dist = dist_list[random.randint(0, 6)]
		location_theta = plus_theta_fn(theta0, theta_list[i])
		location_dist = dist_list[j]
		x1 = x0 + location_dist * math.cos(location_theta)
		y1 = y0 + location_dist * math.sin(location_theta)

		left_pixel = path_finder.point_to_pixel(left_pose)
		right_pixel = path_finder.point_to_pixel((x1, y1))

		# check the line
		flag = rrt.line_check(left_pixel, right_pixel, free)
		if not flag:
			print('j = {}, obstacle'.format(j))
		else:
			diff_theta_idx = random.randint(0, len(diff_theta_list)-1)
			diff_theta = diff_theta_list[diff_theta_idx]
			theta1 = plus_theta_fn(theta0, diff_theta)
			right_pose = [x1, y1, theta1]

			current_pose = right_pose
			pos, orn = func_pose2posAndorn(current_pose, mapper_scene2z[scene_name])
			env.robot.reset_new_pose(pos, orn)
			obs, _, _, _ = env.step(4)
			obs_rgb = obs['rgb_filled']
			cv2.imwrite('{}/right_img_dist_{}_theta_{}.png'.format(file_addr, j, diff_theta_idx), obs_rgb[:,:,::-1])

			right_pose_list.append(right_pose)

## plot the pose graph
file_addr = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs'
img_name = 'temp.jpg'
print('img_name = {}'.format(img_name))
## plot the poses
free = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt/free.png'.format(scene_name), 1)
rows, cols, _ = free.shape
plt.imshow(free)
pose = left_pose
x, y = path_finder.point_to_pixel((pose[0], pose[1]))
theta = pose[2]
plt.arrow(x, y, cos(theta), sin(theta), color='b', \
	overhang=1, head_width=0.1, head_length=0.15, width=0.001)

for m in range(len(right_pose_list)):
	pose = right_pose_list[m]
	x, y = path_finder.point_to_pixel((pose[0], pose[1]))
	theta = pose[2]
	plt.arrow(x, y, cos(theta), sin(theta), color='r', \
		overhang=1, head_width=0.1, head_length=0.15, width=0.001)

plt.axis([0, cols, 0, rows])
plt.xticks([])
plt.yticks([])
plt.savefig('{}/{}'.format(file_addr, img_name), bbox_inches='tight', dpi=(400))
plt.close()




'''
for i in range(20):
	print('i = {}'.format(i))
	len_traj = random.randint(50, 100)
	poses, actions = func_sampleTraj(len_traj)
	num_poses = len(poses)
	left_pose = poses[0]

	for j in range(10):
		right_pose_idx = random.randint(10, num_poses-1)
		right_pose = poses[right_pose_idx]

		left_pixel = path_finder.point_to_pixel(left_pose)
		right_pixel = path_finder.point_to_pixel(right_pose)
		# check the line
		flag = rrt.line_check(left_pixel, right_pixel, free)
		if not flag:
			print('j = {}, obstacle'.format(j))

		## check pose angle difference should be below 90 degree


		## if larger than 90 degree, resample angle for right_pose


		## draw the pose onto maps


		if flag:
			break
'''
