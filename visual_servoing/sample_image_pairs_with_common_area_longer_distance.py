import sys
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/rrt')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing')
import rrt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from math import cos, sin, pi
from util import func_pose2posAndorn, plus_theta_fn, minus_theta_fn
from util_visual_servoing import get_train_test_scenes, get_mapper, create_folder, get_mapper_scene2points, sample_gt_dense_correspondences, get_mapper_scene2points_longer

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

mapper_scene2points = get_mapper_scene2points_longer()
Train_Scenes, Test_Scenes = get_train_test_scenes()
mapper_scene2z = get_mapper()

## create theta list from -45 degree to 45 degree gap 15 degree
## create distance list from 0.5 to 3.0 gap 0.5
theta_list = []
for i in range(7):
    theta = -math.pi/4 + i * math.pi/12
    theta_list.append(theta)
dist_list = [0.5*i for i in range(1, 11)]

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

#mapper_dist = {0:5, 1:10, 2:15, 3:20, 4:25, 5:30}
mapper_dist = {0:5, 1:10, 2:15, 3:20, 4:25, 5:30, 6:35, 7:40, 8:45, 9:50}
mapper_theta = {0:-45, 1:-30, 2:-15, 3:0, 4:15, 5:30, 6:45}

def main(scene_idx):
    scene_name = Test_Scenes[scene_idx]
    #scene_file_addr = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_test/{}'.format(scene_name)
    scene_file_addr = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_test_longer_dist/{}'.format(scene_name)

    #scene_name = Train_Scenes[scene_idx]
    #scene_file_addr = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_train/{}'.format(scene_name)
    create_folder(scene_file_addr)

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

    def get_obs(current_pose):
        pos, orn = func_pose2posAndorn(current_pose, mapper_scene2z[scene_name])
        env.robot.reset_new_pose(pos, orn)
        obs, _, _, _ = env.step(4)
        obs_rgb = obs['rgb_filled']
        obs_depth = obs['depth']
        return obs_rgb.copy(), obs_depth.copy()

    left_pose_list = mapper_scene2points[scene_name]
    right_pose_list = []
    #for p_idx, p in enumerate(left_pose_list):
    for p_idx in range(0, 1):
        p = left_pose_list[p_idx]
        list_whole = []
        x0, y0, theta0 = p
        left_pose = [x0, y0, theta0]

        point_file_addr = '{}/point_{}'.format(scene_file_addr, p_idx)
        create_folder(point_file_addr)
        current_pose = left_pose
        left_rgb, left_depth = get_obs(current_pose)
        cv2.imwrite('{}/left_img.png'.format(point_file_addr), left_rgb[:,:,::-1])
        np.save('{}/left_img_depth.npy'.format(point_file_addr), left_depth)
        ## add left_img to list_whole
        current_dict = {}
        current_dict['img_name'] = 'left_img'
        current_dict['pose'] = left_pose
        list_whole.append(current_dict)

        for i in range(len(theta_list)):
            if i == 0 or i == 6:
                len_dist_list = 2
            elif i == 1 or i == 5:
                len_dist_list = 3
            elif i == 2 or i == 4:
                len_dist_list = 10#4
            elif i == 3:
                len_dist_list = 10#6
            print('len_dist_list = {}'.format(len_dist_list))
            for j in range(len_dist_list):

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
                    for diff_theta_idx in range(len(diff_theta_list)):
                        diff_theta = diff_theta_list[diff_theta_idx]
                        theta1 = plus_theta_fn(theta0, diff_theta)
                        right_pose = [x1, y1, theta1]

                        current_pose = right_pose
                        right_rgb, right_depth = get_obs(current_pose)

                        ## check if there is common space between left img and right img
                        kp1, kp2 = sample_gt_dense_correspondences (left_depth, right_depth, left_pose, 
                            right_pose, gap=32, focal_length=128, resolution=256, start_pixel=31)
                        if kp1.shape[1] > 2:
                            cv2.imwrite('{}/right_img_dist_{}_theta_{}_heading_{}.png'.format(point_file_addr, 
                                mapper_dist[j], mapper_theta[i], mapper_theta[diff_theta_idx]), right_rgb[:,:,::-1])
                            np.save('{}/right_img_dist_{}_theta_{}_heading_{}_depth.npy'.format(point_file_addr, 
                                mapper_dist[j], mapper_theta[i], mapper_theta[diff_theta_idx]), right_depth)
                            right_pose_list.append(right_pose)
                            ## add right_img to list_whole
                            current_dict = {}
                            current_dict['img_name'] = 'right_img_dist_{}_theta_{}_heading_{}'.format(mapper_dist[j], mapper_theta[i], mapper_theta[diff_theta_idx])
                            current_dict['pose'] = right_pose
                            list_whole.append(current_dict)
                        else:
                            print('No common space')

        ## save list_whole
        np.save('{}/point_{}_poses.npy'.format(scene_file_addr, p_idx), list_whole)

    # plot the pose graph
    pose_file_addr = '{}'.format(scene_file_addr)
    img_name = '{}_sampled_poses.jpg'.format(scene_name)
    print('img_name = {}'.format(img_name))
    ## plot the poses
    free = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt/free.png'.format(scene_name), 1)
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

if __name__ == "__main__":
    #for i in range(len(Train_Scenes)):
    #    main(i)
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--scene_idx', type=int, default=0)
    args = parser.parse_args()
    main(args.scene_idx)
