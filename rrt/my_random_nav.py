#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym, logging
from mpi4py import MPI
#from gibson.envs.ant_env import AntNavigateEnv
from gibson.envs.husky_env import HuskyNavigateEnv
from gibson.envs.mobile_robots_env import TurtlebotNavigateEnv
from baselines.common import set_global_seeds
from gibson.utils import pposgd_sensor, pposgd_simple
from gibson.utils import cnn_policy, mlp_policy
import baselines.common.tf_util as U
import datetime
from baselines import logger
import os.path as osp
import random
import cv2
from gibson.core.render.profiler import Profiler
import time
import numpy as np
import skimage.io
from transforms3d.euler import euler2quat
import math

def load_obj(fn):
    verts = []
    faces = []
    with open(fn) as f:
        for line in f:
            if line[:2] == 'v ':
                verts.append(list(map(float, line.strip().split()[1:4])))
            if line[:2] == 'f ':
                face = [int(item.split('/')[0]) for item in line.strip().split()[-3:]]
                faces.append(face)
    verts = np.array(verts)
    faces = np.array(faces) - 1
    return verts, faces

scene_name = 'Allensville'
base_path = 'my_code/rrt/obs_Allensville_train'
#scene_name = 'Cosmos'
#base_path = 'my_code/hand_manipulate/obs_Cosmos_5000_withText' # used for testing code
maximum_steps = 100

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'my_random_navigate.yaml')
print(config_file)

#env = TurtlebotNavigateEnv(config=config_file)
env = HuskyNavigateEnv(config=config_file, gpu_count = 1)
print(env.observation_space)
print(env.action_space)

def callback(obs_t, obs_tp1, rew, done, info):
	return [obs_t, obs_tp1, rew, done, info]

#play(env, zoom=4)
transpose=True
zoom=4
callback=None
keys_to_action=None

obs_s = env.observation_space
#assert type(obs_s) == gym.spaces.box.Box
#assert len(obs_s.shape) == 2 or (len(obs_s.shape) == 3 and obs_s.shape[2] in [1,3])

if keys_to_action is None:
    if hasattr(env, 'get_keys_to_action'):
        keys_to_action = env.get_keys_to_action()
    elif hasattr(env.unwrapped, 'get_keys_to_action'):
        keys_to_action = env.unwrapped.get_keys_to_action()
relevant_keys = set(sum(map(list, keys_to_action.keys()),[]))

pressed_keys = []
running = True
env_done = True

record_num = 0
record_total = 0
obs = env.reset()
do_restart = False
last_keys = []              ## Prevent overacting
obs, _, _, info = env.step(4)

#'/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/Allensville/mesh_z_up.obj'
def quatToXYZW(orn, seq='xyzw'):
    """Convert quaternion from XYZW (pybullet convention) to arbitrary sequence
    """
    assert len(seq) == 4 and 'x' in seq and 'y' in seq and 'z' in seq and 'w' in seq, \
        "Quaternion sequence {} is not valid, please double check.".format(seq)
    inds = [seq.index('x'), seq.index('y'), seq.index('z'), seq.index('w')]
    return orn[inds]
## read the point file
## for the first 5 points
## reset pose and take obs

z = 0.5
#points = np.load('my_code/rrt/points.npy')
#poses = np.load('my_code/rrt/poses.npy')
poses = np.load('my_code/rrt/poses_large.npy')

#for i in range(len(points)-1):
for i in range(len(poses)):
    #p1 = points[i]
    #p2 = points[i+1]
    pose = poses[i]
    #pos = [p1[0], p1[1], z]
    pos = [pose[0], pose[1], z]
    #theta = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
    theta = pose[2]
    orn = quatToXYZW(euler2quat(0, 0, theta), 'wxyz')
    env.robot.reset_new_pose(pos, orn)
    #self.robot_body.reset_orientation(quatToXYZW(euler2quat(*self.config["initial_orn"]), 'wxyz'))
    #self.robot_body.reset_position(self.config["initial_pos"])
    for j in range(20):
        obs, _, _, info = env.step(4)
    obs_rgb = obs['rgb_filled']
    cv2.imwrite(base_path + '/' + str(i) + '_rgb.jpg', obs_rgb[:, :, ::-1])

