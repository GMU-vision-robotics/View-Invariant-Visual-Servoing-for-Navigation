import sys
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)


import my_code.rrt.rrt as rrt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from math import cos, sin, pi, ceil
from my_code.CVPR_workshop.util import func_pose2posAndorn, plus_theta_fn, minus_theta_fn
from my_code.visual_servoing.util_visual_servoing import get_train_test_scenes, get_mapper, create_folder, get_mapper_scene2points, sample_gt_dense_correspondences

#'''
## For Gibson Env
import gym, logging
from mpi4py import MPI
from gibson.envs.husky_env import HuskyNavigateEnv
from baselines import logger
import skimage.io
from transforms3d.euler import euler2quat
#'''

traj_id = 2
scene_name = 'Wainscott'
mapper_scene2z = get_mapper()

## rrt functions
## first figure out how to sample points from rrt graph
rrt_directory = 'gibson/assets/dataset/{}_for_rrt'.format(scene_name)
path_finder = rrt.PathFinder(rrt_directory)
path_finder.load()
num_nodes = len(path_finder.nodes_x)

## GibsonEnv_old setup
config_file = os.path.join('my_code/CVPR_workshop', 'env_yamls', '{}_navigate.yaml'.format(scene_name))
env = HuskyNavigateEnv(config=config_file, gpu_idx=0)
obs = env.reset() ## this line is important otherwise there will be an error like 'AttributeError: 'HuskyNavigateEnv' object has no attribute 'potential''

def get_obs(current_pose):
    pos, orn = func_pose2posAndorn(current_pose, mapper_scene2z[scene_name])
    env.robot.reset_new_pose(pos, orn)
    obs, _, _, _ = env.step(4)
    obs_rgb = obs['rgb_filled']
    obs_depth = obs['depth']
    return obs_rgb.copy(), obs_depth.copy()

dict_poses = np.load('my_code/rrt/topo_path_obs/{}_traj_{}_topo_poses.npy'.format(scene_name, traj_id), allow_pickle=True).item()
for i in range(1):
    print('i = {}'.format(i))
    poses = dict_poses[i]
    len_traj = len(poses)
    num_rows = ceil(len_traj/10)
    num_cols = 10

    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5*11, 5*num_rows))
    for j, pose in enumerate(poses):
        print('j = {}'.format(j))
        left_rgb, _ = get_obs(pose)
        r = j // 10
        c = j % 10
        ax[r][c].imshow(left_rgb)
        ax[r][c].get_xaxis().set_visible(False)
        ax[r][c].get_yaxis().set_visible(False)


    fig.tight_layout()
    #plt.show()
    fig.savefig('{}/{}_traj_{}_obs.png'.format('my_code/rrt/topo_path_obs', scene_name, traj_id))
    plt.close()


'''
# plot the pose graph
pose_file_addr = '{}'.format(scene_file_addr)
img_name = '{}_sampled_poses.jpg'.format(scene_name)
print('img_name = {}'.format(img_name))
## plot the poses
free = cv2.imread('/home/yimeng/Datasets/GibsonEnv_old/gibson/assets/dataset/{}_for_rrt/free.png'.format(scene_name), 1)
rows, cols, _ = free.shape
plt.imshow(free)
for m in range(len(left_pose_list)):
    pose = left_pose_list[m]
    x, y = path_finder.point_to_pixel((pose[0], pose[1]))
    theta = pose[2]
    plt.arrow(x, y, cos(theta), sin(theta), color='r', \
        overhang=1, head_width=0.1, head_length=0.15, width=0.001)
for m in range(len(right_pose_list)):
    pose = right_pose_list[m]
    x, y = path_finder.point_to_pixel((pose[0], pose[1]))
    theta = pose[2]
    plt.arrow(x, y, cos(theta), sin(theta), color='b', \
        overhang=1, head_width=0.1, head_length=0.15, width=0.001)
plt.axis([0, cols, 0, rows])
plt.xticks([])
plt.yticks([])
plt.savefig('{}/{}'.format(pose_file_addr, img_name), bbox_inches='tight', dpi=(400))
plt.close()
'''


