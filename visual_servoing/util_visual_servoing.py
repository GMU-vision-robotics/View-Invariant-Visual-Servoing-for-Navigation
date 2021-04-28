import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as LA
from transforms3d.axangles import mat2axangle, axangle2mat
import cv2
from cyvlfeat.sift import sift
import os
from math import pi, cos, sin, atan2, sqrt
import os, sys
sys.path.append('/home/yimeng/Datasets/GibsonEnv_old/my_code/CVPR_workshop')
from my_code.CVPR_workshop.util import plus_theta_fn, minus_theta_fn
import random

def compute_sift_keypoints(img):
	gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
	kp, des = sift(gray_img, compute_descriptor=True)
	## keypoint frame [y, x], [row, col]
	return kp, des.astype(np.float32)

def get_train_test_scenes():
	Train_Scenes = [ 'Uvalda',  'Collierville', 'Cosmos', 'Hanson', 'Klickitat', 'Lakeville', 'McDade', 
	'Mifflinburg',  'Newfields', 'Onaga', 'Pinesdale', 'Shelbyville', 'Stockman', 'Wiconisco']
	Test_Scenes = ['Allensville', 'Forkland', 'Darden', 'Hiteman', 'Tolstoy', 'Wainscott']
	return Train_Scenes, Test_Scenes

def get_mapper():
	mapper_scene2z = {}
	mapper_scene2z['Allensville'] = 0.47
	mapper_scene2z['Uvalda'] = 0.681
	mapper_scene2z['Darden'] = 3.40
	mapper_scene2z['Collierville'] = 0.46
	mapper_scene2z['Cosmos'] = 0.65
	mapper_scene2z['Forkland'] = 0.35
	mapper_scene2z['Hanson'] = 0.35
	mapper_scene2z['Hiteman'] = 0.40
	mapper_scene2z['Klickitat'] = 0.40
	mapper_scene2z['Lakeville'] = 0.50
	mapper_scene2z['Tolstoy'] = 3.50
	mapper_scene2z['Wainscott'] = 1.0
	mapper_scene2z['McDade'] = 3.16
	mapper_scene2z['Mifflinburg'] = 0.417
	mapper_scene2z['Muleshoe'] = 1.9
	mapper_scene2z['Newfields'] = 0.492
	mapper_scene2z['Noxapater'] = 0.523
	mapper_scene2z['Onaga'] = 0.46
	mapper_scene2z['Pinesdale'] = 0.406
	mapper_scene2z['Pomaria'] = 0.492
	mapper_scene2z['Shelbyville'] = 0.691
	mapper_scene2z['Stockman'] = 3.30
	mapper_scene2z['Wiconisco'] = 0.517
	return mapper_scene2z

def get_mapper_scene2points_old():
	mapper_scene2points = {}
	mapper_scene2points['Allensville'] = [(6.0, 6.5, -pi/2), (6.5, 3.0, pi/2), (4.0, 6.0, -pi/2), (4.0, 2.0, pi/2), (7.0, 6.5, -3.0/4*pi), (5.75, 2.0, pi), (1.75, 7.25, -pi/2), (1.25, 5.0, -pi/2), (1.0, 2.25, pi/2), (0, 0, pi/6), (4.25, 2.0, -pi/2), (4.25, 0, pi/2), (2.5, 2.0, -3.0/4*pi), (6.0, 1.5, 0.0), (8.0, 1.5, pi), (1.5, 2.5, pi), (5.25, 2.0, pi/4), (3.25, 7.5, -pi/2), (3.75, 5.5, pi/2), (6.75, 6.25, pi), (6.0, 5.75, pi), (1.25, 3.5, -pi/4), (3.0, 2.0, 3.0/4*pi), (1.5, 5.0, pi), (-0.25, 5.0, 0.0)]
	mapper_scene2points['Uvalda'] = [(-5.5, -3.0, pi/4), (2.5, 8.0, -pi/2), (-3.5, -3.5, pi*3/4), (-0.5, -1.75, 0), (-3, 4.25, pi*3/4), (-2.25, 3.75, 0), (-1, 2.5, -pi/4), (0.5, -4, pi/2)]
	mapper_scene2points['Darden'] = [(-9.5, 0.5, -3/4*pi), (-7.0, -2.0, pi/2), (-2.5, 1.0, -pi/4), (-9.75, 1.75, -pi/2), (-3.0, -2.5, pi*3/4), (0.0, -0.5, pi*3/4), (0, 2.75, pi)]
	mapper_scene2points['Collierville'] = [(0.8, 2.0, pi), (-2.5, 1.5, -pi/4), (0.0, 4.5, -pi/2), (-3.0, 3.5, -pi/4), (1.0, -1.5, 3.0/4*pi), (-3.0, 3.5, -pi/3), (-2.25, 5.0, 0)]
	mapper_scene2points['Cosmos'] = [(0.0, 12.0, 0), (0.0, -1.5, pi/2), (2.0, 3.0, 0.0), (3.75, 10.5, pi*3/4), (4.5, 6.75, pi), (0.75, 7.5, -pi/2), (0.75, 1.5, pi)]
	mapper_scene2points['Forkland'] = [(-5.25, 2.0, -pi/4), (1.0, -1.0, 3/4*pi), (-4.5, 1.0, 0), (-3, -0.75, -pi/4), (-8.0, 1.25, 0), (-5.25, 3.0, -pi/2), (-6.5, 0.25, -pi/4), (-6.5, -2.5, 0), (2.5, -0.5, pi), (-1, 1, pi/3)]
	mapper_scene2points['Hanson'] = [(-3.0, 8.0, -pi/2), (0.25, -2.0, 2/3*pi), (-3.6, 0.0, 0), (-4.25, 8.25, 0), (-3.25, 4.0, pi/4), (0.75, 4.0, pi*2/3), (0.0, 2.5, -pi/2), (-3.5, -0.5, -pi/2)]
	mapper_scene2points['Hiteman'] = [(7.0, 1.7, 0), (10.0, -0.5, pi/2), (4.0, 1.75, 0), (4.25, 1.0, -pi/4), (1.75, 3.2, pi*5/6), (2.0, 1.0, -pi/4), (-0.5, 2.5, 0)]
	mapper_scene2points['Klickitat'] = [(-2.0, -10.0, 3/4*pi), (1, -2.5, -pi/2), (-4.5, -3.0, -pi/2), (-3, -1.5, 3/4*pi), (-2.5, -0.75, pi), (-3.5, -8.0, pi*3/4)]
	mapper_scene2points['Lakeville'] = [(-6, -0.5, -3/4*pi), (-9.5, -8.0, pi/4), (-13, -7.5, 3/4*pi), (-10, 1.5, -pi/4), (-13, -10, pi*3/4), (-5, -7.5, pi), (-6, -0.5, pi/4), (0.5, 0.0, pi), (-2, -8, pi/2)]
	mapper_scene2points['McDade'] = [(-7.5, -15, pi/6), (-2.5, 0.5, -pi/4), (-1, -1.5, pi*3/4), (-1.7, -7.0, pi/2), (-1.5, -9.0, 0), (-0.5, -12.5, pi*3/4)]
	mapper_scene2points['Mifflinburg'] = [(0.5, -1.0, 5/6*pi), (0.2, 3.5, pi/2), (-2.0, 1.0, pi/2), (0.0, 6.25, pi/2), (-3.0, 7.5, -pi/2), (-4.5, 6.0, 0.0), (-4.25, 3.5, 0), (-1.25, 4.5, -pi*3/4), (0.5, 0.5, pi)]
	mapper_scene2points['Newfields'] = [(-3.5, 8.0, -3/4*pi), (-7.5, 3.25, 0), (2.5, 1.0, pi/2), (-3.5, 6.0, pi), (-2.5, 3.75, pi/2), (-3.5, 3.5, -pi*3/4), (-4, -0.5, pi/2), (-2.5, 4.0, 0), (0.25, 4.0, pi/2), (2.25, 6.0, -pi/2), (-7, 7.75, -pi/4)]
	mapper_scene2points['Tolstoy'] = [(-3.5, 2.5, -pi/6), (-3, 5.5, pi/2), (0.0, 12.0, pi), (0.0, 11.5, -pi/2), (1.75, 5.5, pi*3/4), (-4, 6.5, pi), (-6.5, 1.5, pi/2)]
	mapper_scene2points['Wainscott'] = [(-2.5, 1.5, -pi/4), (-0.5, 9.5, 3/4*pi), (-3.5, -1.25, 0), (-1.5, -6.0, pi/2), (-0.75, 12.5, -pi/2), (1.0, 10.0, 0.0), (3.5, 2.5, -pi/4), (3.5, 6.5, 0)]
	
	mapper_scene2points['Onaga'] = [(-3.5, 2.5, pi/4), (-3.5, 4.0, 0), (0.5, -0.5, pi), (-1.5, 7.0, -pi/2), (-3.5, 5.5, -pi/4), (-3.0, 1.0, -pi/4), (-2.5, 0.5, pi), (-2.5, -1.25, pi)]
	mapper_scene2points['Pinesdale'] = [(-1.0, 20.5, -pi/2), (-0.75, 10.5, pi/2), (-1.75, 20.0, 0), (0.5, 15.5, pi/3), (-0.5, 16.5, pi),(-2.0, 10.0, pi), (-1.5, 10.0, 0), (-3.0, 12.5, pi)]
	mapper_scene2points['Shelbyville'] = [(3.0, -1.5, 0), (1.5, 1.5, pi/2), (-4.25, 8.0, 0), (2.75, 0.5, pi/2), (-4.0, 4.5, 0), (-0.5, 9.5, -pi/2), (-0.75, 5.0, -pi/2), (-4.25, 2.0, pi/2), (-3, -2.5, pi), (-1, -2, -pi/4), (-1.5, 0.25, pi), (4.5, -0.5, -pi/2), (5.25, 0.75, pi/2), (2.75, 5.5, -pi/2), (2.5, 6.5, 0), (3.5, 6.5, pi/2), (5.0, 9.5, pi)]
	mapper_scene2points['Stockman'] = [(2.0, -1.5, pi), (-1, -8.0, 3.0/4*pi), (-2.5, -8.5, pi/2), (-2.5, -8.5, -pi/2), (-2.5, -0.5, pi/2), (-2.5, -0.5, pi), (-2.5, -0.5, -pi/2), (-3.5, -8, pi), (-2.5, -1, 0)]
	mapper_scene2points['Wiconisco'] = [(-3, -1.5, 3.0/4*pi), (1.5, 2.5, 0), (0.5, 3.0, -3.0/4*pi), (1.5, 7.5, 0), (1.5, 9.5, pi/3), (3.5, 10.5, -pi*2/3), (6.5, 7.25, pi), (5.75, 7.5, pi/2), (3.75, 8.0, -pi/2), (2.5, 4.5, 0), (4.5, 4.5, -pi/2), (6.0, 0.75, pi), (1.25, 0.5, pi), (-3.75, 1.5, -pi*3/4)]

	mapper_scene2points['Noxapater'] = [(-1.5, 1.5, -pi/2), (-3.0, -3.0, 0), (4.5, 1.0, pi), (3.0, -2.0, -3.0/4*pi), (7.0, -4.5, 3.0/4*pi)]
	return mapper_scene2points

## remove Allensville's points from 24 to 9
def get_mapper_scene2points():
	mapper_scene2points = {}
	mapper_scene2points['Allensville'] = [(6.0, 6.5, -pi/2), (6.5, 3.0, pi/2), (4.0, 6.0, -pi/2), (4.0, 2.0, pi/2), (7.0, 6.5, -3.0/4*pi), (5.75, 2.0, pi), (1.75, 7.25, -pi/2), (1.25, 5.0, -pi/2), (1.0, 2.25, pi/2)]
	mapper_scene2points['Uvalda'] = [(-5.5, -3.0, pi/4), (2.5, 8.0, -pi/2), (-3.5, -3.5, pi*3/4), (-0.5, -1.75, 0), (-3, 4.25, pi*3/4), (-2.25, 3.75, 0), (-1, 2.5, -pi/4), (0.5, -4, pi/2)]
	mapper_scene2points['Darden'] = [(-9.5, 0.5, -3/4*pi), (-7.0, -2.0, pi/2), (-2.5, 1.0, -pi/4), (-9.75, 1.75, -pi/2), (-3.0, -2.5, pi*3/4), (0.0, -0.5, pi*3/4), (0, 2.75, pi)]
	mapper_scene2points['Collierville'] = [(0.8, 2.0, pi), (-2.5, 1.5, -pi/4), (0.0, 4.5, -pi/2), (-3.0, 3.5, -pi/4), (1.0, -1.5, 3.0/4*pi), (-3.0, 3.5, -pi/3), (-2.25, 5.0, 0)]
	mapper_scene2points['Cosmos'] = [(0.0, 12.0, 0), (0.0, -1.5, pi/2), (2.0, 3.0, 0.0), (3.75, 10.5, pi*3/4), (4.5, 6.75, pi), (0.75, 7.5, -pi/2), (0.75, 1.5, pi)]
	mapper_scene2points['Forkland'] = [(-5.25, 2.0, -pi/4), (1.0, -1.0, 3/4*pi), (-4.5, 1.0, 0), (-3, -0.75, -pi/4), (-8.0, 1.25, 0), (-5.25, 3.0, -pi/2), (-6.5, 0.25, -pi/4), (-6.5, -2.5, 0), (2.5, -0.5, pi), (-1, 1, pi/3)]
	mapper_scene2points['Hanson'] = [(-3.0, 8.0, -pi/2), (0.25, -2.0, 2/3*pi), (-3.6, 0.0, 0), (-4.25, 8.25, 0), (-3.25, 4.0, pi/4), (0.75, 4.0, pi*2/3), (0.0, 2.5, -pi/2), (-3.5, -0.5, -pi/2)]
	mapper_scene2points['Hiteman'] = [(7.0, 1.7, 0), (10.0, -0.5, pi/2), (4.0, 1.75, 0), (4.25, 1.0, -pi/4), (1.75, 3.2, pi*5/6), (2.0, 1.0, -pi/4), (-0.5, 2.5, 0)]
	mapper_scene2points['Klickitat'] = [(-2.0, -10.0, 3/4*pi), (1, -2.5, -pi/2), (-4.5, -3.0, -pi/2), (-3, -1.5, 3/4*pi), (-2.5, -0.75, pi), (-3.5, -8.0, pi*3/4)]
	mapper_scene2points['Lakeville'] = [(-6, -0.5, -3/4*pi), (-9.5, -8.0, pi/4), (-13, -7.5, 3/4*pi), (-10, 1.5, -pi/4), (-13, -10, pi*3/4), (-5, -7.5, pi), (-6, -0.5, pi/4), (0.5, 0.0, pi), (-2, -8, pi/2)]
	mapper_scene2points['McDade'] = [(-7.5, -15, pi/6), (-2.5, 0.5, -pi/4), (-1, -1.5, pi*3/4), (-1.7, -7.0, pi/2), (-1.5, -9.0, 0), (-0.5, -12.5, pi*3/4)]
	mapper_scene2points['Mifflinburg'] = [(0.5, -1.0, 5/6*pi), (0.2, 3.5, pi/2), (-2.0, 1.0, pi/2), (0.0, 6.25, pi/2), (-3.0, 7.5, -pi/2), (-4.5, 6.0, 0.0), (-4.25, 3.5, 0), (-1.25, 4.5, -pi*3/4), (0.5, 0.5, pi)]
	mapper_scene2points['Newfields'] = [(-3.5, 8.0, -3/4*pi), (-7.5, 3.25, 0), (2.5, 1.0, pi/2), (-3.5, 6.0, pi), (-2.5, 3.75, pi/2), (-3.5, 3.5, -pi*3/4), (-4, -0.5, pi/2), (-2.5, 4.0, 0), (0.25, 4.0, pi/2), (2.25, 6.0, -pi/2), (-7, 7.75, -pi/4)]
	mapper_scene2points['Tolstoy'] = [(-3.5, 2.5, -pi/6), (-3, 5.5, pi/2), (0.0, 12.0, pi), (0.0, 11.5, -pi/2), (1.75, 5.5, pi*3/4), (-4, 6.5, pi), (-6.5, 1.5, pi/2)]
	mapper_scene2points['Wainscott'] = [(-2.5, 1.5, -pi/4), (-0.5, 9.5, 3/4*pi), (-3.5, -1.25, 0), (-1.5, -6.0, pi/2), (-0.75, 12.5, -pi/2), (1.0, 10.0, 0.0), (3.5, 2.5, -pi/4), (3.5, 6.5, 0)]
	
	mapper_scene2points['Onaga'] = [(-3.5, 2.5, pi/4), (-3.5, 4.0, 0), (0.5, -0.5, pi), (-1.5, 7.0, -pi/2), (-3.5, 5.5, -pi/4), (-3.0, 1.0, -pi/4), (-2.5, 0.5, pi), (-2.5, -1.25, pi)]
	mapper_scene2points['Pinesdale'] = [(-1.0, 20.5, -pi/2), (-0.75, 10.5, pi/2), (-1.75, 20.0, 0), (0.5, 15.5, pi/3), (-0.5, 16.5, pi),(-2.0, 10.0, pi), (-1.5, 10.0, 0), (-3.0, 12.5, pi)]
	mapper_scene2points['Shelbyville'] = [(3.0, -1.5, 0), (1.5, 1.5, pi/2), (-4.25, 8.0, 0), (2.75, 0.5, pi/2), (-4.0, 4.5, 0), (-0.5, 9.5, -pi/2), (-0.75, 5.0, -pi/2), (-4.25, 2.0, pi/2), (-3, -2.5, pi), (-1, -2, -pi/4), (-1.5, 0.25, pi), (4.5, -0.5, -pi/2), (5.25, 0.75, pi/2), (2.75, 5.5, -pi/2), (2.5, 6.5, 0), (3.5, 6.5, pi/2), (5.0, 9.5, pi)]
	mapper_scene2points['Stockman'] = [(2.0, -1.5, pi), (-1, -8.0, 3.0/4*pi), (-2.5, -8.5, pi/2), (-2.5, -8.5, -pi/2), (-2.5, -0.5, pi/2), (-2.5, -0.5, pi), (-2.5, -0.5, -pi/2), (-3.5, -8, pi), (-2.5, -1, 0)]
	mapper_scene2points['Wiconisco'] = [(-3, -1.5, 3.0/4*pi), (1.5, 2.5, 0), (0.5, 3.0, -3.0/4*pi), (1.5, 7.5, 0), (1.5, 9.5, pi/3), (3.5, 10.5, -pi*2/3), (6.5, 7.25, pi), (5.75, 7.5, pi/2), (3.75, 8.0, -pi/2), (2.5, 4.5, 0), (4.5, 4.5, -pi/2), (6.0, 0.75, pi), (1.25, 0.5, pi), (-3.75, 1.5, -pi*3/4)]

	mapper_scene2points['Noxapater'] = [(-1.5, 1.5, -pi/2), (-3.0, -3.0, 0), (4.5, 1.0, pi), (3.0, -2.0, -3.0/4*pi), (7.0, -4.5, 3.0/4*pi)]
	return mapper_scene2points

## remove Allensville's points from 24 to 9
def get_mapper_scene2points_longer():
	mapper_scene2points = {}
	mapper_scene2points['Allensville'] = [(6.0, 7.5, -pi/2), (6.5, 3.0, pi/2), (4.0, 6.0, -pi/2), (4.0, 2.0, pi/2), (7.0, 6.5, -3.0/4*pi), (5.75, 2.0, pi), (1.75, 7.25, -pi/2), (1.25, 5.0, -pi/2), (1.0, 2.25, pi/2)]
	mapper_scene2points['Uvalda'] = [(-5.5, -3.0, pi/4), (2.5, 8.0, -pi/2), (-3.5, -3.5, pi*3/4), (-0.5, -1.75, 0), (-3, 4.25, pi*3/4), (-2.25, 3.75, 0), (-1, 2.5, -pi/4), (0.5, -4, pi/2)]
	mapper_scene2points['Darden'] = [(-9.5, 0.5, -3/4*pi), (-7.0, -2.0, pi/2), (-2.5, 1.0, -pi/4), (-9.75, 1.75, -pi/2), (-3.0, -2.5, pi*3/4), (0.0, -0.5, pi*3/4), (0, 2.75, pi)]
	mapper_scene2points['Collierville'] = [(0.8, 2.0, pi), (-2.5, 1.5, -pi/4), (0.0, 4.5, -pi/2), (-3.0, 3.5, -pi/4), (1.0, -1.5, 3.0/4*pi), (-3.0, 3.5, -pi/3), (-2.25, 5.0, 0)]
	mapper_scene2points['Cosmos'] = [(0.0, 12.0, 0), (0.0, -1.5, pi/2), (2.0, 3.0, 0.0), (3.75, 10.5, pi*3/4), (4.5, 6.75, pi), (0.75, 7.5, -pi/2), (0.75, 1.5, pi)]
	mapper_scene2points['Forkland'] = [(-5.25, 2.0, -pi/4), (1.0, -1.0, 3/4*pi), (-4.5, 1.0, 0), (-3, -0.75, -pi/4), (-8.0, 1.25, 0), (-5.25, 3.0, -pi/2), (-6.5, 0.25, -pi/4), (-6.5, -2.5, 0), (2.5, -0.5, pi), (-1, 1, pi/3)]
	mapper_scene2points['Hanson'] = [(-3.0, 8.0, -pi/2), (0.25, -2.0, 2/3*pi), (-3.6, 0.0, 0), (-4.25, 8.25, 0), (-3.25, 4.0, pi/4), (0.75, 4.0, pi*2/3), (0.0, 2.5, -pi/2), (-3.5, -0.5, -pi/2)]
	mapper_scene2points['Hiteman'] = [(7.0, 1.7, 0), (10.0, -0.5, pi/2), (4.0, 1.75, 0), (4.25, 1.0, -pi/4), (1.75, 3.2, pi*5/6), (2.0, 1.0, -pi/4), (-0.5, 2.5, 0)]
	mapper_scene2points['Klickitat'] = [(-2.0, -10.0, 3/4*pi), (1, -2.5, -pi/2), (-4.5, -3.0, -pi/2), (-3, -1.5, 3/4*pi), (-2.5, -0.75, pi), (-3.5, -8.0, pi*3/4)]
	mapper_scene2points['Lakeville'] = [(-6, -0.5, -3/4*pi), (-9.5, -8.0, pi/4), (-13, -7.5, 3/4*pi), (-10, 1.5, -pi/4), (-13, -10, pi*3/4), (-5, -7.5, pi), (-6, -0.5, pi/4), (0.5, 0.0, pi), (-2, -8, pi/2)]
	mapper_scene2points['McDade'] = [(-7.5, -15, pi/6), (-2.5, 0.5, -pi/4), (-1, -1.5, pi*3/4), (-1.7, -7.0, pi/2), (-1.5, -9.0, 0), (-0.5, -12.5, pi*3/4)]
	mapper_scene2points['Mifflinburg'] = [(0.5, -1.0, 5/6*pi), (0.2, 3.5, pi/2), (-2.0, 1.0, pi/2), (0.0, 6.25, pi/2), (-3.0, 7.5, -pi/2), (-4.5, 6.0, 0.0), (-4.25, 3.5, 0), (-1.25, 4.5, -pi*3/4), (0.5, 0.5, pi)]
	mapper_scene2points['Newfields'] = [(-3.5, 8.0, -3/4*pi), (-7.5, 3.25, 0), (2.5, 1.0, pi/2), (-3.5, 6.0, pi), (-2.5, 3.75, pi/2), (-3.5, 3.5, -pi*3/4), (-4, -0.5, pi/2), (-2.5, 4.0, 0), (0.25, 4.0, pi/2), (2.25, 6.0, -pi/2), (-7, 7.75, -pi/4)]
	mapper_scene2points['Tolstoy'] = [(-3.5, 2.5, -pi/6), (-3, 5.5, pi/2), (0.0, 12.0, pi), (0.0, 11.5, -pi/2), (1.75, 5.5, pi*3/4), (-4, 6.5, pi), (-6.5, 1.5, pi/2)]
	mapper_scene2points['Wainscott'] = [(-2.5, 1.5, -pi/4), (-0.5, 9.5, 3/4*pi), (-3.5, -1.25, 0), (-1.5, -6.0, pi/2), (-0.75, 12.5, -pi/2), (1.0, 10.0, 0.0), (3.5, 2.5, -pi/4), (3.5, 6.5, 0)]
	
	mapper_scene2points['Onaga'] = [(-3.5, 2.5, pi/4), (-3.5, 4.0, 0), (0.5, -0.5, pi), (-1.5, 7.0, -pi/2), (-3.5, 5.5, -pi/4), (-3.0, 1.0, -pi/4), (-2.5, 0.5, pi), (-2.5, -1.25, pi)]
	mapper_scene2points['Pinesdale'] = [(-1.0, 20.5, -pi/2), (-0.75, 10.5, pi/2), (-1.75, 20.0, 0), (0.5, 15.5, pi/3), (-0.5, 16.5, pi),(-2.0, 10.0, pi), (-1.5, 10.0, 0), (-3.0, 12.5, pi)]
	mapper_scene2points['Shelbyville'] = [(3.0, -1.5, 0), (1.5, 1.5, pi/2), (-4.25, 8.0, 0), (2.75, 0.5, pi/2), (-4.0, 4.5, 0), (-0.5, 9.5, -pi/2), (-0.75, 5.0, -pi/2), (-4.25, 2.0, pi/2), (-3, -2.5, pi), (-1, -2, -pi/4), (-1.5, 0.25, pi), (4.5, -0.5, -pi/2), (5.25, 0.75, pi/2), (2.75, 5.5, -pi/2), (2.5, 6.5, 0), (3.5, 6.5, pi/2), (5.0, 9.5, pi)]
	mapper_scene2points['Stockman'] = [(2.0, -1.5, pi), (-1, -8.0, 3.0/4*pi), (-2.5, -8.5, pi/2), (-2.5, -8.5, -pi/2), (-2.5, -0.5, pi/2), (-2.5, -0.5, pi), (-2.5, -0.5, -pi/2), (-3.5, -8, pi), (-2.5, -1, 0)]
	mapper_scene2points['Wiconisco'] = [(-3, -1.5, 3.0/4*pi), (1.5, 2.5, 0), (0.5, 3.0, -3.0/4*pi), (1.5, 7.5, 0), (1.5, 9.5, pi/3), (3.5, 10.5, -pi*2/3), (6.5, 7.25, pi), (5.75, 7.5, pi/2), (3.75, 8.0, -pi/2), (2.5, 4.5, 0), (4.5, 4.5, -pi/2), (6.0, 0.75, pi), (1.25, 0.5, pi), (-3.75, 1.5, -pi*3/4)]

	mapper_scene2points['Noxapater'] = [(-1.5, 1.5, -pi/2), (-3.0, -3.0, 0), (4.5, 1.0, pi), (3.0, -2.0, -3.0/4*pi), (7.0, -4.5, 3.0/4*pi)]
	return mapper_scene2points

def create_folder (folder_name):
	flag_exist = os.path.isdir(folder_name)
	if not flag_exist:
		print('{} folder does not exist, so create one.'.format(folder_name))
		os.makedirs(folder_name)
		#os.makedirs(os.path.join(test_case_folder, 'observations'))
	else:
		print('{} folder already exists, so do nothing.'.format(folder_name))

def get_mapper_dist_theta_heading():
	mapper_dist = {'5': 0.5, '10': 1.0, '15': 1.5, '20': 2.0, '25': 2.5, '30': 3.0}
	mapper_theta = {'0': 0.0, '15': pi/12, '30': pi/6, '45': pi/4, '-15': -pi/12, '-30': -pi/6, '-45': -pi/4}
	mapper_heading = {'0': 0.0, '15': pi/12, '30': pi/6, '45': pi/4, '-15': -pi/12, '-30': -pi/6, '-45': -pi/4}
	return mapper_dist, mapper_theta, mapper_heading

def get_pose_from_name(img_name, ref_pose):
	dist = img_name.split('_')[3]
	theta = img_name.split('_')[5]
	heading = img_name.split('_')[-1][:-4]
	mapper_dist, mapper_theta, mapper_heading = get_mapper_dist_theta_heading()
	location_dist = mapper_dist[dist]
	theta = mapper_theta[theta]
	heading = mapper_heading[heading]

	x0, y0, theta0 = ref_pose
	location_theta = plus_theta_fn(theta0, theta)
	x1 = x0 + location_dist * math.cos(location_theta)
	y1 = y0 + location_dist * math.sin(location_theta)
	theta1 = plus_theta_fn(theta0, heading)
	right_pose = [x1, y1, theta1]
	return right_pose

def toAffinity(f):
	T = f[0:2]
	scale = f[2]
	theta = f[3]
	A = np.zeros((3, 3), np.float32)
	A[0, 0] = scale * math.cos(theta)
	A[0, 1] = scale * math.sin(theta) * (-1)
	A[1, 0] = scale * math.sin(theta)
	A[1, 1] = scale * math.cos(theta)
	A[0:2, 2] = T
	A[2, 0] = 0
	A[2, 1] = 0
	A[2, 2] = 1
	return A

## this function computes coordinates normalization matrix of input keypoints
## input keypoints shaped: 2 x num_keypoings
def centering(x):
	T = np.zeros((3,3), np.float32)
	T[0, 0] = 1
	T[1, 1] = 1
	T[0:2, 2] = (-1) * np.mean(x[0:2, :], axis=1)
	T[2, :] = [0, 0, 1]
	x = np.matmul(T, x)
	std1 = x[0,:].std(axis=0)
	std2 = x[1,:].std(axis=0)

	# at least one pixel apart to avoid numerical problems
	std1 = np.maximum(std1, 1)
	std2 = np.maximum(std2, 1)

	S = np.array([[1.0/std1, 0, 0], [0, 1.0/std2, 0], [0, 0, 1]], np.float32)
	C = np.matmul(S, T)
	return C

def find_good_matches(kp1, kp2, np_matches, good):
	kp1_good = np.zeros((len(good), 4), np.float32)
	kp2_good = np.zeros((len(good), 4), np.float32)
	kp1_good = kp1[np_matches[good, 1]]
	kp2_good = kp2[np_matches[good, 2]]
	kp1_good = kp1_good.transpose()
	kp2_good = kp2_good.transpose()

	opts_tolerance1 = 20
	opts_tolerance2 = 15
	opts_tolerance3 = 8
	opts_minInliers = 6
	opts_numRefinementIterations = 5

	numMatches = len(good)
	inliers = {}
	H = {}

	x1 = kp1_good[0:2, :]
	x2 = kp2_good[0:2, :]

	x1hom = np.zeros((3, x1.shape[1]), np.float32)
	x1hom[0:2, :] = x1
	x1hom[2, :] = 1
	x2hom = np.zeros((3, x2.shape[1]), np.float32)
	x2hom[0:2, :] = x2
	x2hom[2, :] = 1

	for m in range(numMatches):
		for t in range(opts_numRefinementIterations):
			if t == 0:
				A1 = toAffinity(kp1_good[:, m])
				A2 = toAffinity(kp2_good[:, m])
				# A2 = H21 * A1
				H21 = np.matmul(A2, np.linalg.inv(A1))
				# project x1hom onto x2 using H21*x1hom
				x1p = np.matmul(H21[0:2, :], x1hom)
				tol = opts_tolerance1
			elif t <= 3:
				# affinity
				#(H21*x1hom=x2) == (xA=B). For python, A.T*x.T=B.T. A is x1hom, B is x2. 
				H21_temp = np.linalg.lstsq(x1hom[:, inliers[m]].T, x2[:, inliers[m]].T)[0].T
				# project x1hom onto x2
				x1p = np.matmul(H21_temp[0:2, :], x1hom)
				H21 = np.zeros((3,3), np.float32)
				H21[0:2, :] = H21_temp
				H21[2, :] = [0, 0, 1]
				tol = opts_tolerance2
			else:
				# homography
				# get homogeneous coords of inlier matches
				x1in = x1hom[:, inliers[m]]
				x2in = x2hom[:, inliers[m]]

				S1 = centering(x1in)
				S2 = centering(x2in)
				x1c = np.matmul(S1, x1in)
				x2c = np.matmul(S2, x2in)

				r, c = x1c.shape
				M = np.zeros((r*3, c*2), np.float32)
				M[0:r, 0:c] = x1c
				M[r:2*r, c:2*c] = x1c
				M[2*r:3*r, 0:c] = x1c * (-x2c)[0,:]
				M[2*r:3*r, c:2*c] = x1c * (-x2c)[1,:]

				H21, D, _ = np.linalg.svd(M, full_matrices=False)
				H21 = np.reshape(H21[:, -1], (3,3)).T
				H21 = np.matmul(np.matmul(np.linalg.inv(S2),H21), S1)
				H21 = H21 / H21[-1,-1]
				## x2 = H x1
				## project x1 onto camera2's frame through homography matrix H21
				x1phom = np.matmul(H21, x1hom)
				x1p = np.zeros((2, x1phom.shape[1]), np.float32)
				x1p[0,:] = x1phom[0,:] / x1phom[2,:]
				x1p[1,:] = x1phom[1,:] / x1phom[2,:]
				tol = opts_tolerance3

			# compute square distance between x2 and x1p
			dist2 = np.sum(np.power((x2 - x1p), 2), axis=0)
			# find matches with distance satisfies tolerance
			inliers[m] = (dist2 < (tol^2)).nonzero()[0]
			H[m] = H21
			if len(inliers[m]) < opts_minInliers:
				break
			if len(inliers[m]) > 0.7 * numMatches:
				break

	scores = []
	for k, v in inliers.items():
		scores.append(len(v))
	best = scores.index(max(scores))
	best_inliers = inliers[best]
	H = np.linalg.inv(H[best])

	return best_inliers, H

def find_closest_orthonormal_matrix(m):
	u, _, v = LA.svd(m)
	return u.dot(v.T)

def compute_rotation(m):
	d,v = LA.eig(m)
	[dr] = np.logical_and(d.imag==0,d.real!=0).nonzero()
	#print('dr = {}'.format(dr))
	#dr = dr.item()
	dr = dr[0]
	u = v[:, dr].real
	if dr == 0:
		theta = np.arccos(d[1].real)
	else:
		theta = np.arccos(d[0].real)
	#print('theta = {}'.format(theta))
	if np.isnan(theta):
		m = find_closest_orthonormal_matrix(m) 
		d,v = LA.eig(m)
		[dr] = np.logical_and(d.imag==0,d.real!=0).nonzero()
		dr = dr.item()
		u = v[:, dr].real
		if dr == 0:
			theta = np.arccos(d[1].real)
		else:
			theta = np.arccos(d[0].real)
	## resolve sign ambiguity
	dif = m - axangle2mat(u, theta)
	epsilon = np.ones((3,3))*0.0001
	if not np.all(np.logical_and(-epsilon <= dif, dif <= epsilon)):
		theta = -theta
	return theta

def get_number_of_good_matches(img1, img2):
	kp1, des1 = compute_sift_keypoints(img1)
	kp2, des2 = compute_sift_keypoints(img2)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)

	np_matches = np.zeros((len(matches), 6), dtype=np.float32)
	for i in range(len(matches)):
		m, n = matches[i]
		np_matches[i, 0] = m.distance
		np_matches[i, 1] = m.queryIdx
		np_matches[i, 2] = m.trainIdx
		np_matches[i, 3] = n.distance
		np_matches[i, 4] = n.queryIdx
		np_matches[i, 5] = n.trainIdx

	# ratio test as per Lowe's paper
	good = [] ## contains good point index
	for idx in range(np_matches.shape[0]):
		m_dis, m_queryIdx, _, n_dis, _, _ = np_matches[idx, :]
		if m_dis < 0.8 * n_dis:
			good.append(idx)
	np_matches = np_matches.astype(np.int16)

	## find good matches
	best_inliers, H_test = find_good_matches(kp1, kp2, np_matches, good)
	good = np.array(good)
	good = good[best_inliers]
	return len(best_inliers)

def get_num_matches_and_write_image(img1, img2, img_name):
	kp1, des1 = compute_sift_keypoints(img1)
	kp2, des2 = compute_sift_keypoints(img2)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)

	np_matches = np.zeros((len(matches), 6), dtype=np.float32)
	for i in range(len(matches)):
		m, n = matches[i]
		np_matches[i, 0] = m.distance
		np_matches[i, 1] = m.queryIdx
		np_matches[i, 2] = m.trainIdx
		np_matches[i, 3] = n.distance
		np_matches[i, 4] = n.queryIdx
		np_matches[i, 5] = n.trainIdx

	# ratio test as per Lowe's paper
	good = [] ## contains good point index
	for idx in range(np_matches.shape[0]):
		m_dis, m_queryIdx, _, n_dis, _, _ = np_matches[idx, :]
		if m_dis < 0.8 * n_dis:
			good.append(idx)
	np_matches = np_matches.astype(np.int16)

	## find good matches
	best_inliers, H_test = find_good_matches(kp1, kp2, np_matches, good)
	good = np.array(good)
	good = good[best_inliers]

	## compute homography matrix
	kp1 = kp1[np_matches[good, 1]].T[0:2,:]
	kp2 = kp2[np_matches[good, 2]].T[0:2,:]

	## convert keypoints to homocoordinates
	x1hom = np.zeros((3, kp1.shape[1]), np.float32)
	x1hom[0:2, :] = kp1
	x1hom[2, :] = 1
	x2hom = np.zeros((3, kp2.shape[1]), np.float32)
	x2hom[0:2, :] = kp2
	x2hom[2, :] = 1

	# normalize keypoing coordinates
	T1 = centering(x1hom)
	T2 = centering(x2hom)
	x1norm = T1.dot(x1hom)
	x2norm = T2.dot(x2hom)

	## compute normalized homography matrix H tilde
	r, c = x1norm.shape
	A = np.zeros((r*3, c*2), np.float32) # shape: 9 x (num_kps x 2)
	A[0:r, 0:c] = x1norm
	A[r:2*r, c:2*c] = x1norm
	A[2*r:3*r, 0:c] = x1norm * (-x2norm)[0,:]
	A[2*r:3*r, c:2*c] = x1norm * (-x2norm)[1,:]

	H21, D, _ = np.linalg.svd(A, full_matrices=False)
	## use H21[:, -1] to choose the smallest column of U corresponding to smallest singular value
	H21 = np.reshape(H21[:, -1], (3,3)).T

	## convert normalized homography matrix into unnormalized homography matrix
	H21 = np.matmul(np.matmul(np.linalg.inv(T2), H21), T1)
	## divide the homography matrix from its bottom right value so that its bottom right value is 1
	H21 = H21 / H21[-1, -1]

	## visualize the projected keypoints through homography matrix
	#'''
	x1phom = H21.dot(x1hom)
	x1p = np.zeros((2, x1phom.shape[1]), np.float32)
	x1p[0,:] = x1phom[0,:] / x1phom[2,:]
	x1p[1,:] = x1phom[1,:] / x1phom[2,:]

	img_combined = np.concatenate((img1, img2), axis=1)
	plt.imshow(img_combined)
	plt.plot(kp1[1, :], kp1[0, :], 'ro')
	plt.plot(x1p[1, :]+256, x1p[0, :], 'ro')
	for i in range(len(good)):
		plt.plot([kp1[1, :], x1p[1, :]+256], 
			[kp1[0, :], x1p[0, :]], 'ro-')
	plt.plot(kp1[1, :], kp1[0, :], 'bo')
	plt.plot(kp2[1, :]+256, kp2[0, :], 'bo')
	for i in range(len(good)):
		plt.plot([kp1[1, :], kp2[1, :]+256], 
			[kp1[0, :], kp2[0, :]], 'bo-')
	plt.savefig(img_name, bbox_inches='tight')
	plt.close()

	return len(best_inliers)

def estimate_essential_matrix(kp1, kp2):
	## read in kp1 as xim1 and read in kp2 as xim2
	xim1 = np.ones((3, kp1.shape[1]))
	xim2 = np.ones((3, kp2.shape[1]))
	xim1[:2, :] = kp1
	xim2[:2, :] = kp2

	## backproject xim1, xim2 through K to get xr1, xr2
	K = np.array([128,0,0, 0,128,0, 0,0,1])
	K = K.reshape((3,3))
	xr1 = LA.inv(K).dot(xim1)
	xr2 = LA.inv(K).dot(xim2)

	## start the essentialDiscrete Alg
	p = xr1
	q = xr2

	_, NPOINTS = p.shape

	## set up matrix A such that A*[v1,v2,v3,s1,s2,s3,s4,s5,s6].T = 0
	A = np.zeros((NPOINTS, 9))

	if NPOINTS < 9:
		assert('Too few correspondeces')

	for i in range(NPOINTS):
		A[i, :] = np.kron(q[:, i], p[:, i]).T
	r = LA.matrix_rank(A)

	if r < 8:
		T0 = 0
		R = np.identity(3)

	U, S, V = np.linalg.svd(A)
	V = V.T
	## pick the eigenvector corresponding to the smallest eigenvalue
	e = V[:, -1]
	e = np.around(e, decimals=4)
	## essential matrix
	E = e.reshape(3, 3)

	## then four possibilities are
	Rzp = np.array([[0,-1,0], [1,0,0], [0,0,1]]) # rotation by pi/2
	Rzn = np.array([[0,1,0], [-1,0,0], [0,0,1]]) # rotation by -pi/2

	U, S, V = LA.svd(E)
	V = V.T
	S = np.diag((1,1,0))
	detu = LA.det(U)
	detv = LA.det(V)
	if detu < 0 and detv < 0:
		U = -U
		V = -V
	elif detu < 0 and detv > 0:
		S1 = Rzp.dot(S)
		U = (-U).dot(axangle2mat([S1[2, 1], S1[0, 2], S1[1, 0]], math.pi)).dot(Rzp)
		V = V.dot(Rzp)
	elif detu > 0 and detv < 0:
		S1 = Rzp.dot(S)
		U = U.dot(axangle2mat([S1[2,1], S1[0,2], S1[1,0]], math.pi)).dot(Rzp)
		V = (-V).dot(Rzp)

	## initialize R, Th and t
	R  = np.empty((3, 3, 4))
	Th = np.empty((3, 3, 4))
	t  = np.empty((3, 4))
	omega = np.empty((3, 4))
	theta = np.empty((4))

	R[:, :, 0] = U.dot(Rzp.T).dot(V.T)
	Th[:, :, 0] = U.dot(Rzp).dot(S).dot(U.T)
	t[:, 0] = np.array([-Th[1, 2, 0], Th[0, 2, 0], -Th[0, 1, 0]]).T
	omega[:, 0], theta[0] = mat2axangle(R[:,:,0])

	R[:, :, 1] = U.dot(Rzn.T).dot(V.T)
	Th[:, :, 1] = U.dot(Rzn).dot(S).dot(U.T)
	t[:, 1] = np.array([-Th[1,2,1], Th[0,2,1], -Th[0,1,1]]).T
	omega[:, 1], theta[1] = mat2axangle(R[:,:,1])

	U, S, V = LA.svd(-E)
	V = V.T
	S = np.diag((1,1,0))
	detu = LA.det(U)
	detv = LA.det(V)
	if detu < 0 and detv < 0:
		U = -U
		V = -V
	elif detu < 0 and detv > 0:
		S1 = Rzp.dot(S)
		U = (-U).dot(axangle2mat([S1[2, 1], S1[0, 2], S1[1, 0]], math.pi)).dot(Rzp)
		V = V.dot(Rzp)
	elif detu > 0 and detv < 0:
		S1 = Rzp.dot(S)
		U = U.dot(axangle2mat([S1[2,1], S1[0,2], S1[1,0]], math.pi)).dot(Rzp)
		V = (-V).dot(Rzp)

	R[:, :, 2] = U.dot(Rzp.T).dot(V.T)
	Th[:, :, 2] = U.dot(Rzp).dot(S).dot(U.T)
	t[:, 2] = np.array([-Th[1, 2, 2], Th[0, 2, 2], -Th[0, 1, 2]]).T
	omega[:, 2], theta[2] = mat2axangle(R[:, :, 2])

	R[:, :, 3] = U.dot(Rzn.T).dot(V.T)
	Th[:, :, 3] = U.dot(Rzn).dot(S).dot(U.T)
	t[:, 3] = np.array([-Th[1,2,3], Th[0,2,3], -Th[0,1,3]]).T
	omega[:, 3], theta[3] = mat2axangle(R[:,:,3])

	## triple product of three vectors (a x b) c
	## corresponds to the volume spanned by them
	def triple_product(a, b, c):
		volume = c[0] * (a[1]*b[2] - b[1]*a[2]) + \
				c[1] * (a[2]*b[0] - b[2]*a[0]) + \
				c[2] * (a[0]*b[1] - b[0]*a[1])
		return volume
	## build skew symmetric matrix
	def skew(x):
		return np.array([[0, -x[2], x[1]], [x[2], 0, x[0]], [x[1], x[0], 0]])

	## compute volume which ahs to be positive if the two scales have the same sign
	## and then check whether one of the scale is positive
	posdepth = np.zeros(4)
	for i in range(4):
		volume = np.empty(NPOINTS)
		alpha1 = np.empty(NPOINTS)
		alpha2 = np.empty(NPOINTS)
		for j in range(NPOINTS):
			volume[j] = triple_product(t[:, i], R[:,:,i].dot(p[:,j]), Th[:,:,i].dot(q[:,j]))

			alpha1[j] = -(skew(q[:, j]).dot(t[:, i])).T.dot(skew(q[:,j]).dot(R[:,:,i]).dot(p[:,j]))/(LA.norm(skew(q[:,j]).dot(t[:,i]), ord=2))**2
			alpha2[j] = (skew(R[:,:,i].dot(p[:,j])).dot(q[:,j])).T.dot(skew(R[:,:,i].dot(p[:,j])).dot(t[:,i]))/(LA.norm(skew(R[:,:,i].dot(p[:,j])).dot(q[:,j]), ord=2))**2

		vol = np.sum(np.sign(volume))
		depth1 = np.sum(np.sign(alpha1))
		depth2 = np.sum(np.sign(alpha2))

		posdepth[i] = vol + depth1

	val, index = posdepth.max(0), posdepth.argmax(0)
	T0 = t[:, index]
	R0 = R[:, :, index]
	return R0, T0

def estimate_rotation_from_two_views(img1, img2):
	kp1, des1 = compute_sift_keypoints(img1)
	kp2, des2 = compute_sift_keypoints(img2)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)

	np_matches = np.zeros((len(matches), 6), dtype=np.float32)
	for i in range(len(matches)):
		m, n = matches[i]
		np_matches[i, 0] = m.distance
		np_matches[i, 1] = m.queryIdx
		np_matches[i, 2] = m.trainIdx
		np_matches[i, 3] = n.distance
		np_matches[i, 4] = n.queryIdx
		np_matches[i, 5] = n.trainIdx

	# ratio test as per Lowe's paper
	good = [] ## contains good point index
	for idx in range(np_matches.shape[0]):
		m_dis, m_queryIdx, _, n_dis, _, _ = np_matches[idx, :]
		if m_dis < 0.8 * n_dis:
			good.append(idx)
	np_matches = np_matches.astype(np.int16)

	## find good matches
	best_inliers, H_test = find_good_matches(kp1, kp2, np_matches, good)
	good = np.array(good)
	good = good[best_inliers]

	## compute homography matrix
	kp1 = kp1[np_matches[good, 1]].T[0:2,:]
	kp2 = kp2[np_matches[good, 2]].T[0:2,:]

	rotation_matrix, _ = estimate_essential_matrix(kp1, kp2)
	return compute_rotation(rotation_matrix)

def detect_correspondences(img1, img2):
	kp1, des1 = compute_sift_keypoints(img1)
	kp2, des2 = compute_sift_keypoints(img2)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)

	np_matches = np.zeros((len(matches), 6), dtype=np.float32)
	for i in range(len(matches)):
		m, n = matches[i]
		np_matches[i, 0] = m.distance
		np_matches[i, 1] = m.queryIdx
		np_matches[i, 2] = m.trainIdx
		np_matches[i, 3] = n.distance
		np_matches[i, 4] = n.queryIdx
		np_matches[i, 5] = n.trainIdx

	# ratio test as per Lowe's paper
	good = [] ## contains good point index
	for idx in range(np_matches.shape[0]):
		m_dis, m_queryIdx, _, n_dis, _, _ = np_matches[idx, :]
		if m_dis < 0.8 * n_dis:
			good.append(idx)
	np_matches = np_matches.astype(np.int16)

	## find good matches
	best_inliers, H_test = find_good_matches(kp1, kp2, np_matches, good)
	good = np.array(good)
	good = good[best_inliers]

	## compute homography matrix
	kp1 = kp1[np_matches[good, 1]].T[0:2,:]
	kp2 = kp2[np_matches[good, 2]].T[0:2,:]
	return kp1, kp2

def detect_correspondences_and_descriptors(img1, img2):
	kp1, des1 = compute_sift_keypoints(img1)
	kp2, des2 = compute_sift_keypoints(img2)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)

	np_matches = np.zeros((len(matches), 6), dtype=np.float32)
	for i in range(len(matches)):
		m, n = matches[i]
		np_matches[i, 0] = m.distance
		np_matches[i, 1] = m.queryIdx
		np_matches[i, 2] = m.trainIdx
		np_matches[i, 3] = n.distance
		np_matches[i, 4] = n.queryIdx
		np_matches[i, 5] = n.trainIdx

	# ratio test as per Lowe's paper
	good = [] ## contains good point index
	for idx in range(np_matches.shape[0]):
		m_dis, m_queryIdx, _, n_dis, _, _ = np_matches[idx, :]
		if m_dis < 0.8 * n_dis:
			good.append(idx)
	np_matches = np_matches.astype(np.int16)

	## find good matches
	best_inliers, H_test = find_good_matches(kp1, kp2, np_matches, good)
	good = np.array(good)
	good = good[best_inliers]

	## compute homography matrix
	kp1 = kp1[np_matches[good, 1]].T[0:2,:]
	kp2 = kp2[np_matches[good, 2]].T[0:2,:]
	des1 = des1[np_matches[good, 1]]
	des2 = des2[np_matches[good, 2]]

	return kp1, kp2, des1, des2

## for each descriptor in kp1, find its correspondence in img2
def detect_correspondences_for_fixed_kps(des1, img2):
	kp2, des2 = compute_sift_keypoints(img2)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)

	np_matches = np.zeros((len(matches), 6), dtype=np.float32)
	for i in range(len(matches)):
		m, n = matches[i]
		np_matches[i, 0] = m.distance
		np_matches[i, 1] = m.queryIdx
		np_matches[i, 2] = m.trainIdx
		np_matches[i, 3] = n.distance
		np_matches[i, 4] = n.queryIdx
		np_matches[i, 5] = n.trainIdx

	# ratio test as per Lowe's paper
	good = [] ## contains good point index
	for idx in range(np_matches.shape[0]):
		m_dis, m_queryIdx, _, n_dis, _, _ = np_matches[idx, :]
		if m_dis < 0.8 * n_dis:
			good.append(idx)

	np_matches = np_matches.astype(np.int16)
	kp2 = kp2[np_matches[good, 2]].T[0:2,:]

	return kp2, np_matches[good, 1]



## estimate depth between correspondeces kp1 and kp2 given pose transformation
def estimate_depth(kp1, kp2, current_pose, next_pose):
	## compute the rotation and translation from camera1 pose (current pose) to camera2 pose (next pose)
	'''
	def compute_tx_tz_theta(current_pose, next_pose):
		x1, y1, theta1 = next_pose
		x0, y0, theta0 = current_pose
		x_change = x1 - x0
		y_change = y1 - y0
		theta_change = theta1 - theta0
		## look from upper side is different from look from downside
		## So clockwise is the positive direction
		#theta_change = -theta_change
		dist = math.sqrt(x_change**2 + y_change**2)
		theta0_real = atan2(y_change, x_change)
		tx = 0.0
		if abs(theta0_real - theta0) > pi/2:
			tz = -dist
		else:
			tz = dist
		return tx, tz, theta_change
	'''
	def compute_tx_tz_theta(current_pose, next_pose):
		x1, y1, theta1 = next_pose
		x0, y0, theta0 = current_pose
		x_change = x1 - x0
		y_change = y1 - y0
		theta_change = minus_theta_fn(theta0, theta1)
		dist = math.sqrt(x_change**2 + y_change**2)
		theta0_real = atan2(y_change, x_change)
		phi = minus_theta_fn(theta0_real, theta0)
		tz = dist * cos(phi)
		tx = dist * sin(phi)
		
		return tx, tz, theta_change
	
	def svdsolve(A, b):
		#u,s,v = LA.svd(A, full_matrices=False)
		u,s,v = LA.svd(A)
		c = np.dot(u.T,b)
		#w = LA.solve(np.diag(s),c)
		w = np.divide(c[:len(s)],s)
		x = np.dot(v.T,w)
		return x
	
	num_matches = kp1.shape[1]
	## rotation and translation from current frame to next frame
	tx, tz, theta = compute_tx_tz_theta(current_pose, next_pose)
	## rotation matrix from camera1 to camera2
	R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
	## rotation matrix from camera2 to camera1
	temp_m = -R.T.dot(np.array([[tx], [tz]]))
	## rotation and translation from next frame to current frame
	tx, tz = temp_m
	theta = -theta

	lambda_focal = 128.0
	u0 = lambda_focal
	v0 = lambda_focal
	Zs = np.ones(num_matches)
	for i in range(num_matches):
		v, u = kp1[:, i]
		v_prime, u_prime = kp2[:, i]
		#a = lambda_focal*cos(theta) + (u_prime - u0)*sin(theta)
		a = lambda_focal*cos(theta) - (u_prime - u0)*sin(theta)
		#b = lambda_focal*sin(theta) - (u_prime-u0)*cos(theta)
		b = -lambda_focal*sin(theta) - (u_prime-u0)*cos(theta)
		c = lambda_focal*tx - (u_prime-u0)*tz
		#d = -(v_prime-v0) * sin(theta)
		d = (v_prime-v0) * sin(theta)
		e = (v_prime-v0) * cos(theta)
		f = (v_prime - v0) * tz
		## build A
		A = np.zeros((4, 3))
		A[0, 0] = lambda_focal
		A[0, 2] = -(u - u0)
		A[1, 1] = lambda_focal
		A[1, 2] = -(v - v0)
		A[2, 0] = a
		A[2, 2] = b  
		A[3, 0] = d
		A[3, 1] = -lambda_focal
		A[3, 2] = e
		## build b
		b = np.zeros((4, 1))
		b[2, 0] = -c 
		b[3, 0] = -f 
		#x = svdsolve(A, b)
		x, _, _, _ = np.linalg.lstsq(A, b)
		if x[2] < 0:
			Zs[i] = 1.0
		else:
			Zs[i] = x[2]
	return Zs

## estimate depth between correspondeces kp1 and kp2 given pose transformation
def estimate_depth_remove_bad(kp1, kp2, current_pose, next_pose):
	def compute_tx_tz_theta(current_pose, next_pose):
		x1, y1, theta1 = next_pose
		x0, y0, theta0 = current_pose
		x_change = x1 - x0
		y_change = y1 - y0
		theta_change = minus_theta_fn(theta0, theta1)
		dist = math.sqrt(x_change**2 + y_change**2)
		theta0_real = atan2(y_change, x_change)
		phi = minus_theta_fn(theta0_real, theta0)
		tz = dist * cos(phi)
		tx = dist * sin(phi)
		
		return tx, tz, theta_change

	num_matches = kp1.shape[1]
	## rotation and translation from current frame to next frame
	tx, tz, theta = compute_tx_tz_theta(current_pose, next_pose)
	## rotation matrix from camera1 to camera2
	R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
	## rotation matrix from camera2 to camera1
	temp_m = -R.T.dot(np.array([[tx], [tz]]))
	## rotation and translation from next frame to current frame
	tx, tz = temp_m
	theta = -theta

	lambda_focal = 128.0
	u0 = lambda_focal
	v0 = lambda_focal
	Zs = []
	good = []
	for i in range(num_matches):
		v, u = kp1[:, i]
		v_prime, u_prime = kp2[:, i]
		#a = lambda_focal*cos(theta) + (u_prime - u0)*sin(theta)
		a = lambda_focal*cos(theta) - (u_prime - u0)*sin(theta)
		#b = lambda_focal*sin(theta) - (u_prime-u0)*cos(theta)
		b = -lambda_focal*sin(theta) - (u_prime-u0)*cos(theta)
		c = lambda_focal*tx - (u_prime-u0)*tz
		#d = -(v_prime-v0) * sin(theta)
		d = (v_prime-v0) * sin(theta)
		e = (v_prime-v0) * cos(theta)
		f = (v_prime - v0) * tz
		## build A
		A = np.zeros((4, 3))
		A[0, 0] = lambda_focal
		A[0, 2] = -(u - u0)
		A[1, 1] = lambda_focal
		A[1, 2] = -(v - v0)
		A[2, 0] = a
		A[2, 2] = b  
		A[3, 0] = d
		A[3, 1] = -lambda_focal
		A[3, 2] = e
		## build b
		b = np.zeros((4, 1))
		b[2, 0] = -c 
		b[3, 0] = -f 
		#x = svdsolve(A, b)
		x, _, _, _ = np.linalg.lstsq(A, b)
		if x[2] > 0:
			Zs.append(x[2])
			good.append(i)
	return Zs, good

def estimate_depth_from_startView(kp1, kp2, start_pose, current_pose):
	def compute_tx_tz_theta(current_pose, next_pose):
		x1, y1, theta1 = next_pose
		x0, y0, theta0 = current_pose
		x_change = x1 - x0
		y_change = y1 - y0
		theta_change = minus_theta_fn(theta0, theta1)
		dist = math.sqrt(x_change**2 + y_change**2)
		theta0_real = atan2(y_change, x_change)
		phi = minus_theta_fn(theta0_real, theta0)
		tz = dist * cos(phi)
		tx = dist * sin(phi)
		return tx, tz, theta_change
	
	num_matches = kp1.shape[1]
	## rotation and translation from current frame to next frame
	tx, tz, theta = compute_tx_tz_theta(start_pose, current_pose)
	## rotation matrix from camera1 to camera2
	R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
	## rotation matrix from camera2 to camera1
	temp_m = -R.T.dot(np.array([[tx], [tz]]))
	## rotation and translation from next frame to current frame
	tx, tz = temp_m
	theta = -theta

	lambda_focal = 128.0
	u0 = lambda_focal
	v0 = lambda_focal
	Zs = np.ones(num_matches)
	for i in range(num_matches):
		v, u = kp1[:, i]
		v_prime, u_prime = kp2[:, i]
		#a = lambda_focal*cos(theta) + (u_prime - u0)*sin(theta)
		a = lambda_focal*cos(theta) - (u_prime - u0)*sin(theta)
		#b = lambda_focal*sin(theta) - (u_prime-u0)*cos(theta)
		b = -lambda_focal*sin(theta) - (u_prime-u0)*cos(theta)
		c = lambda_focal*tx - (u_prime-u0)*tz
		#d = -(v_prime-v0) * sin(theta)
		d = (v_prime-v0) * sin(theta)
		e = (v_prime-v0) * cos(theta)
		f = (v_prime - v0) * tz
		## build A
		A = np.zeros((4, 3))
		A[0, 0] = lambda_focal
		A[0, 2] = -(u - u0)
		A[1, 1] = lambda_focal
		A[1, 2] = -(v - v0)
		A[2, 0] = a
		A[2, 2] = b  
		A[3, 0] = d
		A[3, 1] = -lambda_focal
		A[3, 2] = e
		## build b
		b = np.zeros((4, 1))
		b[2, 0] = -c 
		b[3, 0] = -f 
		#x = svdsolve(A, b)
		x, _, _, _ = np.linalg.lstsq(A, b)
		Z_in_current_frame = -x[0]*sin(theta) + x[2]*cos(theta) + tz ##-X*sin(theta) + Z*cos(theta) +tz
		if Z_in_current_frame < 0:
			Zs[i] = 1.0
		else:
			Zs[i] = Z_in_current_frame
	return Zs

def compute_new_pose(current_pose, dist=0.1):
	x0, y0, theta0 = current_pose
	x1 = x0 + dist * math.cos(theta0)
	y1 = y0 + dist * math.sin(theta0)
	print('dist * cos(theta) = {}, dist * sin(theta) = {}'.format(dist * math.cos(theta0), dist * math.sin(theta0)))
	return [x1, y1, theta0]

def sample_gt_dense_correspondences_unefficient (current_depth, goal_depth, current_pose, goal_pose, gap=32, focal_length=128, resolution=256, start_pixel=31):
	def dense_correspondence_compute_tx_tz_theta(current_pose, next_pose):
		x1, y1, theta1 = next_pose
		x0, y0, theta0 = current_pose
		phi = atan2(y1-y0, x1-x0)
		gamma = minus_theta_fn(theta0, phi)
		dist = math.sqrt((x1-x0)**2 + (y1-y0)**2)
		tz = dist * cos(gamma)
		tx = -dist * sin(gamma)
		theta_change = (theta1 - theta0)
		#print('dist = {}'.format(dist))
		#print('gamma = {}'.format(gamma))
		#print('theta_change = {}'.format(theta_change))
		return tx, tz, theta_change

	x = [i for i in range(start_pixel, resolution-start_pixel, gap)]
	## densely sample keypoints for current image
	## first axis of kp1 is 'u', second dimension is 'v'
	kp1 = np.empty((2, len(x)*len(x)))
	count = 0
	for j in range(len(x)):
		for i in range(len(x)):
			kp1[0, count] = x[i]
			kp1[1, count] = x[j]
			count += 1
	#print('kp1 = {}'.format(kp1[:, :10]))
	## camera intrinsic matrix
	K = np.array([[focal_length, 0, focal_length], [0, focal_length, focal_length], [0, 0, 1]])
	## expand kp1 from 2 dimensions to 3 dimensions
	kp1_3d = np.ones((3, kp1.shape[1]))
	kp1_3d[:2, :] = kp1

	## backproject kp1_3d through inverse of K and get kp1_3d. x=KX, X is in the camera frame
	## Now kp1_3d still have the third dimension Z to be 1.0. This is the world coordinates in camera frame after projection.
	kp1_3d = LA.inv(K).dot(kp1_3d)

	## backproject kp1_3d into world coords kp1_4d by using gt-depth
	## Now kp1_4d has coords in world frame if camera1 (current) frame coincide with the world frame
	kp1_4d = np.ones((4, kp1.shape[1]))
	for i in range(kp1.shape[1]):
		Z = current_depth[int(kp1[1, i]), int(kp1[0, i])]
		kp1_4d[2, i] = Z
		kp1_4d[0, i] = Z * kp1_3d[0, i]
		kp1_4d[1, i] = Z * kp1_3d[1, i]
	#print('kp1_4d = {}'.format(kp1_4d[:, :10]))

	## first compute the rotation and translation from current frame to goal frame
	tx, tz, theta = dense_correspondence_compute_tx_tz_theta(current_pose, goal_pose)
	bad = []
	for i in range(kp1.shape[1]):
		Z = current_depth[int(kp1[1, i]), int(kp1[0, i])]
		if Z <= tz:
			bad.append(i)
	#print('bad = {}'.format(bad))
	R = np.array([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]])
	T = np.array([tx, 0, tz])
	## then compute the transformation matrix from goal frame to current frame
	## thransformation matrix is the camera2's extrinsic matrix
	transformation_matrix = np.empty((3, 4))
	transformation_matrix[:3, :3] = R.T
	transformation_matrix[:3, 3] = -R.T.dot(T)

	## transform kp1_4d from camera1(current) frame to camera2(goal) frame through transformation matrix
	kp2_3d = transformation_matrix.dot(kp1_4d)
	kp2_3d_cpy = kp2_3d.copy()
	## project kp2_3d into a plane. So that the Z dimension is 1.0
	kp2_3d[0, :] = kp2_3d[0, :] / kp2_3d[2, :]
	kp2_3d[1, :] = kp2_3d[1, :] / kp2_3d[2, :]
	kp2_3d[2, :] = kp2_3d[2, :] / kp2_3d[2, :]
	## pass kp2_3d through intrinsic matrix
	kp2 = K.dot(kp2_3d)
	## give up the last dimension of kp2. Only keep the first and second dimension 'u' and 'v'
	#print('kp2 = {}'.format(kp2[:, :10]))
	kp2 = np.floor(kp2[:2, :])

	good = []
	for i in range(kp1.shape[1]):
		u_prime = kp2[0, i]
		v_prime = kp2[1, i]
		if u_prime < resolution and u_prime >= 0 and v_prime >= 0 and v_prime < resolution:
			good.append(i)
	bad_depth = []
	for i in good:
		goal_Z = goal_depth[int(kp2[1, i]), int(kp2[0, i])]
		if abs(goal_Z - kp2_3d_cpy[2, i]) > 0.1:
			bad_depth.append(i)
	good_remove_bad = []
	for i in good:
		if i not in bad and i not in bad_depth:
				good_remove_bad.append(i)
	kp2 = kp2[::-1, good_remove_bad]
	kp1 = kp1[::-1, good_remove_bad]
	return kp1, kp2

def sample_gt_dense_correspondences (current_depth, goal_depth, current_pose, goal_pose, gap=32, focal_length=128, resolution=256, start_pixel=31, depth_verification=True):
	def dense_correspondence_compute_tx_tz_theta(current_pose, next_pose):
		x1, y1, theta1 = next_pose
		x0, y0, theta0 = current_pose
		phi = atan2(y1-y0, x1-x0)
		gamma = minus_theta_fn(theta0, phi)
		dist = math.sqrt((x1-x0)**2 + (y1-y0)**2)
		tz = dist * cos(gamma)
		tx = -dist * sin(gamma)
		theta_change = (theta1 - theta0)
		#print('dist = {}'.format(dist))
		#print('gamma = {}'.format(gamma))
		#print('theta_change = {}'.format(theta_change))
		return tx, tz, theta_change

	## camera intrinsic matrix
	K = np.array([[focal_length, 0, focal_length], [0, focal_length, focal_length], [0, 0, 1]])
	inv_K = LA.inv(K)
	## first compute the rotation and translation from current frame to goal frame
	tx, tz, theta = dense_correspondence_compute_tx_tz_theta(current_pose, goal_pose)
	R = np.array([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]])
	T = np.array([tx, 0, tz])
	## then compute the transformation matrix from goal frame to current frame
	## thransformation matrix is the camera2's extrinsic matrix
	transformation_matrix = np.empty((3, 4))
	transformation_matrix[:3, :3] = R.T
	transformation_matrix[:3, 3] = -R.T.dot(T)
	## build fundamental matrix
	#fundamental_mat = K.dot(transformation_matrix).dot(LA.inv(K))

	coords_range = range(start_pixel, resolution-start_pixel, gap)
	kp1_x, kp1_y = np.meshgrid(np.array(coords_range), np.array(coords_range))
	kp1_Z = current_depth[kp1_y.reshape(-1), kp1_x.reshape(-1)].reshape((len(coords_range), len(coords_range)))
	kp1_4d = np.ones((len(coords_range), len(coords_range), 4), np.float32)
	kp1_4d[:, :, 0] = kp1_x
	kp1_4d[:, :, 1] = kp1_y
	kp1_4d[:, :, 2] = kp1_Z
	#print('kp1_4d.shape = {}.format()'.format(kp1_4d.shape))
	kp1_4d = np.transpose(kp1_4d, (2, 0, 1)).reshape((4, -1))
	#print('kp1_4d.shape = {}.format()'.format(kp1_4d.shape))
	kp1 = np.floor(kp1_4d[:2, :])
	#print('kp1 = {}'.format(kp1[:, :10]))

	kp1_4d[[0, 1, 3], :] = inv_K.dot(kp1_4d[[0, 1, 3], :])
	kp1_4d[0, :] = kp1_4d[0, :] * kp1_4d[2, :]
	kp1_4d[1, :] = kp1_4d[1, :] * kp1_4d[2, :]
	#print('kp1_4d = {}'.format(kp1_4d[:, :10]))

	kp2_3d = transformation_matrix.dot(kp1_4d)
	kp2_3d = K.dot(kp2_3d)
	kp2_3d[0, :] = kp2_3d[0, :] / kp2_3d[2, :]
	kp2_3d[1, :] = kp2_3d[1, :] / kp2_3d[2, :]
	#print('kp2_3d = {}'.format(kp2_3d[:, :10]))
	kp2 = kp2_3d[:2, :]
	kp2 = np.floor(kp2)
	assert kp1.shape[1] == kp2.shape[1] 

	good = []
	for i in range(kp1_4d.shape[1]):
		u_prime = kp2[0, i]
		v_prime = kp2[1, i]
		if u_prime < resolution and u_prime >= 0 and v_prime >= 0 and v_prime < resolution:
			if (kp1_4d[2, i] > tz and tz >= 0) or (tz < 0):
				if depth_verification:
					goal_Z = goal_depth[int(v_prime), int(u_prime)]
					if abs(goal_Z - kp2_3d[2, i]) <= 0.1:
						good.append(i)
				else:
					good.append(i)

	kp1 = kp1[::-1, good]
	kp2 = kp2[::-1, good]
	return kp1, kp2

def sample_gt_dense_correspondences_in_bbox (current_depth, goal_depth, current_pose, goal_pose, bbox, gap=32, focal_length=128, resolution=256, start_pixel=31, depth_verification=True):
	def dense_correspondence_compute_tx_tz_theta(current_pose, next_pose):
		x1, y1, theta1 = next_pose
		x0, y0, theta0 = current_pose
		phi = atan2(y1-y0, x1-x0)
		gamma = minus_theta_fn(theta0, phi)
		dist = math.sqrt((x1-x0)**2 + (y1-y0)**2)
		tz = dist * cos(gamma)
		tx = -dist * sin(gamma)
		theta_change = (theta1 - theta0)
		#print('dist = {}'.format(dist))
		#print('gamma = {}'.format(gamma))
		#print('theta_change = {}'.format(theta_change))
		return tx, tz, theta_change

	## camera intrinsic matrix
	K = np.array([[focal_length, 0, focal_length], [0, focal_length, focal_length], [0, 0, 1]])
	inv_K = LA.inv(K)
	## first compute the rotation and translation from current frame to goal frame
	tx, tz, theta = dense_correspondence_compute_tx_tz_theta(current_pose, goal_pose)
	R = np.array([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]])
	T = np.array([tx, 0, tz])
	## then compute the transformation matrix from goal frame to current frame
	## thransformation matrix is the camera2's extrinsic matrix
	transformation_matrix = np.empty((3, 4))
	transformation_matrix[:3, :3] = R.T
	transformation_matrix[:3, 3] = -R.T.dot(T)
	## build fundamental matrix
	#fundamental_mat = K.dot(transformation_matrix).dot(LA.inv(K))

	coords_range = range(start_pixel, resolution-start_pixel, gap)
	kp1_x, kp1_y = np.meshgrid(np.array(coords_range), np.array(coords_range))
	kp1_Z = current_depth[kp1_y.reshape(-1), kp1_x.reshape(-1)].reshape((len(coords_range), len(coords_range)))
	kp1_4d = np.ones((len(coords_range), len(coords_range), 4), np.float32)
	kp1_4d[:, :, 0] = kp1_x
	kp1_4d[:, :, 1] = kp1_y
	kp1_4d[:, :, 2] = kp1_Z
	#print('kp1_4d.shape = {}.format()'.format(kp1_4d.shape))
	kp1_4d = np.transpose(kp1_4d, (2, 0, 1)).reshape((4, -1))
	#print('kp1_4d.shape = {}.format()'.format(kp1_4d.shape))
	kp1 = np.floor(kp1_4d[:2, :])
	#print('kp1 = {}'.format(kp1[:, :10]))

	kp1_4d[[0, 1, 3], :] = inv_K.dot(kp1_4d[[0, 1, 3], :])
	kp1_4d[0, :] = kp1_4d[0, :] * kp1_4d[2, :]
	kp1_4d[1, :] = kp1_4d[1, :] * kp1_4d[2, :]
	#print('kp1_4d = {}'.format(kp1_4d[:, :10]))

	kp2_3d = transformation_matrix.dot(kp1_4d)
	kp2_3d = K.dot(kp2_3d)
	kp2_3d[0, :] = kp2_3d[0, :] / kp2_3d[2, :]
	kp2_3d[1, :] = kp2_3d[1, :] / kp2_3d[2, :]
	#print('kp2_3d = {}'.format(kp2_3d[:, :10]))
	kp2 = kp2_3d[:2, :]
	kp2 = np.floor(kp2)
	assert kp1.shape[1] == kp2.shape[1] 

	good = []
	x1, y1, x2, y2 = bbox
	for i in range(kp1_4d.shape[1]):
		u_prime = kp2[0, i]
		v_prime = kp2[1, i]
		if u_prime < x2 and u_prime >= x1 and v_prime >= y1 and v_prime < y2:
			if (kp1_4d[2, i] > tz and tz >= 0) or (tz < 0):
				if depth_verification:
					goal_Z = goal_depth[int(v_prime), int(u_prime)]
					if abs(goal_Z - kp2_3d[2, i]) <= 0.1:
						good.append(i)
				else:
					good.append(i)

	kp1 = kp1[::-1, good]
	kp2 = kp2[::-1, good]
	return kp1, kp2

def sample_gt_correspondences_with_large_displacement (current_depth, goal_depth, current_pose, goal_pose, gap=32, focal_length=128, resolution=256, start_pixel=31):
	kp1, kp2 = sample_gt_dense_correspondences(current_depth, goal_depth, current_pose, goal_pose, gap, focal_length, resolution, start_pixel)
	## pick top 4
	if kp1.shape[1] > 4:
		## compute the top 4 correspondence with largest displacement
		kps_displacement = np.empty((kp1.shape[1]))
		for i in range(kp1.shape[1]):
			v, u = kp1[:, i]
			v_prime, u_prime = kp2[:, i]
			displacement = abs(v_prime - v) + abs(u_prime - u)
			kps_displacement[i] = displacement
		top_4_idx = np.argsort(kps_displacement)[-4:]
		kp1 = kp1[:, top_4_idx]
		kp2 = kp2[:, top_4_idx]
	return kp1, kp2

def sample_gt_correspondences_relativelyDense (current_depth, goal_depth, current_pose, goal_pose, gap=32, focal_length=128, resolution=256, start_pixel=31):
	kp1, kp2 = sample_gt_dense_correspondences(current_depth, goal_depth, current_pose, goal_pose, gap=1, focal_length=128, resolution=256, start_pixel=1)
	'''
	## pick top 4
	if kp1.shape[1] > 4:
		## randomly pick 4 correspondences
		num_kps = kp1.shape[1]
		idx_list = [i for i in range(num_kps)]
		random_4_idx = random.sample(idx_list, 4)
		kp1 = kp1[:, random_4_idx]
		kp2 = kp2[:, random_4_idx]
	'''
	return kp1, kp2

def sample_gt_corner_correspondences (current_depth, goal_depth, current_pose, goal_pose, gap=32, focal_length=128, resolution=256, start_pixel=31):
	def dense_correspondence_compute_tx_tz_theta(current_pose, next_pose):
		x1, y1, theta1 = next_pose
		x0, y0, theta0 = current_pose
		phi = atan2(y1-y0, x1-x0)
		gamma = minus_theta_fn(theta0, phi)
		dist = math.sqrt((x1-x0)**2 + (y1-y0)**2)
		tz = dist * cos(gamma)
		tx = -dist * sin(gamma)
		theta_change = (theta1 - theta0)
		#print('dist = {}'.format(dist))
		#print('gamma = {}'.format(gamma))
		#print('theta_change = {}'.format(theta_change))
		return tx, tz, theta_change

	kp2 = np.array([[start_pixel, start_pixel], [start_pixel, resolution-start_pixel], [resolution-start_pixel, start_pixel], [resolution-start_pixel, resolution-start_pixel]]).T
	K = np.array([[focal_length, 0, focal_length], [0, focal_length, focal_length], [0, 0, 1]])
	## expand kp2 from 2 dimensions to 3 dimensions
	kp2_3d = np.ones((3, kp2.shape[1]))
	kp2_3d[:2, :] = kp2

	kp2_3d = LA.inv(K).dot(kp2_3d)

	kp2_4d = np.ones((4, kp2.shape[1]))
	for i in range(kp2.shape[1]):
		Z = goal_depth[int(kp2[1, i]), int(kp2[0, i])]
		kp2_4d[2, i] = Z
		kp2_4d[0, i] = Z * kp2_3d[0, i]
		kp2_4d[1, i] = Z * kp2_3d[1, i]

	tx, tz, theta = dense_correspondence_compute_tx_tz_theta(current_pose, goal_pose)

	R = np.array([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]])
	T = np.array([tx, 0, tz])
	transformation_matrix = np.empty((3, 4))
	transformation_matrix[:3, :3] = R
	transformation_matrix[:3, 3] = T

	kp1_3d = transformation_matrix.dot(kp2_4d)
	kp1_3d_cpy = kp1_3d.copy()
	kp1_3d[0, :] = kp1_3d[0, :] / kp1_3d[2, :]
	kp1_3d[1, :] = kp1_3d[1, :] / kp1_3d[2, :]
	kp1_3d[2, :] = kp1_3d[2, :] / kp1_3d[2, :]
	kp1 = K.dot(kp1_3d)
	kp1 = np.floor(kp1[:2, :])

	good = []
	for i in range(kp2.shape[1]):
		u_prime = kp1[0, i]
		v_prime = kp1[1, i]
		if u_prime < resolution and u_prime >= 0 and v_prime >= 0 and v_prime < resolution:
			good.append(i)
	bad_depth = []
	for i in good:
		current_Z = current_depth[int(kp1[1, i]), int(kp1[0, i])]
		if abs(current_Z - kp1_3d_cpy[2, i]) > 0.1:
			bad_depth.append(i)
	good_remove_bad = []
	for i in good:
		if i not in bad_depth:
				good_remove_bad.append(i)
	kp2 = kp2[::-1, good_remove_bad]
	kp1 = kp1[::-1, good_remove_bad]

	return kp1, kp2

def sample_gt_dense_correspondences_frow_goalView (current_depth, goal_depth, current_pose, goal_pose, gap=32, focal_length=128, resolution=256, start_pixel=31):
	def dense_correspondence_compute_tx_tz_theta(current_pose, next_pose):
		x1, y1, theta1 = next_pose
		x0, y0, theta0 = current_pose
		phi = atan2(y1-y0, x1-x0)
		gamma = minus_theta_fn(theta0, phi)
		dist = math.sqrt((x1-x0)**2 + (y1-y0)**2)
		tz = dist * cos(gamma)
		tx = -dist * sin(gamma)
		theta_change = (theta1 - theta0)
		#print('dist = {}'.format(dist))
		#print('gamma = {}'.format(gamma))
		#print('theta_change = {}'.format(theta_change))
		return tx, tz, theta_change

	x = [i for i in range(start_pixel, resolution-start_pixel, gap)]
	## densely sample keypoints for current image
	## first axis of kp1 is 'u', second dimension is 'v'
	kp2 = np.empty((2, len(x)*len(x)))
	count = 0
	for i in range(len(x)):
		for j in range(len(x)):
			kp2[0, count] = x[i]
			kp2[1, count] = x[j]
			count += 1

	K = np.array([[focal_length, 0, focal_length], [0, focal_length, focal_length], [0, 0, 1]])
	## expand kp2 from 2 dimensions to 3 dimensions
	kp2_3d = np.ones((3, kp2.shape[1]))
	kp2_3d[:2, :] = kp2

	kp2_3d = LA.inv(K).dot(kp2_3d)

	kp2_4d = np.ones((4, kp2.shape[1]))
	for i in range(kp2.shape[1]):
		Z = goal_depth[int(kp2[1, i]), int(kp2[0, i])]
		kp2_4d[2, i] = Z
		kp2_4d[0, i] = Z * kp2_3d[0, i]
		kp2_4d[1, i] = Z * kp2_3d[1, i]

	tx, tz, theta = dense_correspondence_compute_tx_tz_theta(current_pose, goal_pose)

	R = np.array([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]])
	T = np.array([tx, 0, tz])
	transformation_matrix = np.empty((3, 4))
	transformation_matrix[:3, :3] = R
	transformation_matrix[:3, 3] = T

	kp1_3d = transformation_matrix.dot(kp2_4d)
	kp1_3d_cpy = kp1_3d.copy()
	kp1_3d[0, :] = kp1_3d[0, :] / kp1_3d[2, :]
	kp1_3d[1, :] = kp1_3d[1, :] / kp1_3d[2, :]
	kp1_3d[2, :] = kp1_3d[2, :] / kp1_3d[2, :]
	kp1 = K.dot(kp1_3d)
	kp1 = np.floor(kp1[:2, :])

	good = []
	for i in range(kp2.shape[1]):
		u_prime = kp1[0, i]
		v_prime = kp1[1, i]
		if u_prime < resolution and u_prime >= 0 and v_prime >= 0 and v_prime < resolution:
			good.append(i)
	bad_depth = []
	for i in good:
		current_Z = current_depth[int(kp1[1, i]), int(kp1[0, i])]
		if abs(current_Z - kp1_3d_cpy[2, i]) > 0.1:
			bad_depth.append(i)
	good_remove_bad = []
	for i in good:
		if i not in bad_depth:
				good_remove_bad.append(i)
	kp2 = kp2[::-1, good_remove_bad]
	kp1 = kp1[::-1, good_remove_bad]

	return kp1, kp2


def sample_gt_dense_correspondences_frow_goalView_with_large_displacement (current_depth, goal_depth, current_pose, goal_pose, gap=32, focal_length=128, resolution=256, start_pixel=31):
	kp1, kp2 = sample_gt_dense_correspondences_frow_goalView(current_depth, goal_depth, current_pose, goal_pose, gap, focal_length, resolution, start_pixel)

	if kp1.shape[1] > 4:
		## compute the top 4 correspondence with largest displacement
		kps_displacement = np.empty((kp1.shape[1]))
		for i in range(kp1.shape[1]):
			v, u = kp1[:, i]
			v_prime, u_prime = kp2[:, i]
			displacement = abs(v_prime - v) + abs(u_prime - u)
			kps_displacement[i] = displacement
		top_4_idx = np.argsort(kps_displacement)[-4:]
		kp1 = kp1[:, top_4_idx]
		kp2 = kp2[:, top_4_idx]

	return kp1, kp2

def expand_target_object_img_to_desired_size(img, target_h=256, target_w=256):
	h, w, _ = img.shape
	assert h <= target_h
	assert w <= target_w

	final_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
	left_upper_corner_y = (target_h - h) // 2
	left_upper_corner_x = (target_w - w) // 2
	final_img[left_upper_corner_y:left_upper_corner_y+h, left_upper_corner_x:left_upper_corner_x+w] = img

	return final_img

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

def compute_gt_velocity_through_interaction_matrix (current_depth, goal_depth, current_pose, goal_pose):
	kp1, kp2 = sample_gt_correspondences_with_large_displacement(current_depth, goal_depth, current_pose, goal_pose)
	num_matches = kp1.shape[1]
	kp1_Z = np.empty((3, num_matches))
	kp1_Z[:2, :] = kp1
	for i in range(num_matches):
		kp1_Z[2, i] = current_depth[int(kp1_Z[0, i]), int(kp1_Z[1, i])]
	## build L matrix
	L = build_L_matrix(kp1_Z)
	## updating the projection errors
	e = kp1[::-1,:].flatten('F') - kp2[::-1,:].flatten('F')
	#vc = -0.5*LA.pinv(L).dot(e)
	vc = -LA.pinv(L).dot(e)
	#vx, vz, omegay= 0.5 * vc
	vx, vz, omegay = 0.25*vc
	omegay = -omegay

	## check if we should stop or not
	flag_stop = False
	if num_matches > 0:
		displacement = np.sum(e**2) / num_matches
		if  displacement < 25:
			flag_stop = True
	else:
		flag_stop = True

	return vx, vz, omegay, flag_stop

def compute_velocity_through_correspondences_and_depth (kp1, kp2, current_depth):
	num_matches = kp1.shape[1]
	if num_matches == 0:
		return 0.0, 0.0, 0.0, True

	kp1_Z = np.empty((3, num_matches))
	kp1_Z[:2, :] = kp1
	for i in range(num_matches):
		kp1_Z[2, i] = current_depth[int(kp1_Z[0, i]), int(kp1_Z[1, i])]
	## build L matrix
	L = build_L_matrix(kp1_Z)
	## updating the projection errors
	e = kp1[::-1,:].flatten('F') - kp2[::-1,:].flatten('F')
	#vc = -0.5*LA.pinv(L).dot(e)
	#vx, vz, omegay= 0.5 * vc
	vc = -LA.pinv(L).dot(e)
	vx, vz, omegay = 0.25*vc
	omegay = -omegay

	displacement = 0.0
	## check if we should stop or not
	flag_stop = False
	if num_matches > 0:
		displacement = np.sum(e**2) / num_matches
		if  displacement < 25:
			flag_stop = True
	else:
		flag_stop = True

	return vx, vz, omegay, flag_stop#, displacement

def compute_angular_velocity_through_correspondences(kp1, kp2, thresh=500):
	num_matches = kp1.shape[1]
	if num_matches == 0:
		return 0.0, True

	lambda_focal = 128.0
	num_points = kp1.shape[1]
	u0 = lambda_focal
	v0 = lambda_focal
	L = np.empty((2*num_points, 1))
	for i in range(num_points):
		v, u= kp1[:, i]
		u = u - u0
		v = v - v0
		L[2*i, :]   = np.array([-lambda_focal-u*u/lambda_focal])
		L[2*i+1, :] = np.array([-u*v/lambda_focal])
	## updating the projection errors
	e = kp1[::-1,:].flatten('F') - kp2[::-1,:].flatten('F')
	vc = -LA.pinv(L).dot(e)
	omegay = 0.5*vc[0]
	omegay = -omegay

	## check if we should stop or not
	flag_stop = False
	if num_matches > 0:
		displacement = np.sum(e**2) / num_matches
		if  displacement < thresh:
			flag_stop = True
	else:
		flag_stop = True

	return omegay, flag_stop


## randomly sample 1000 kps
def sample_gt_random_dense_correspondences (current_depth, goal_depth, current_pose, goal_pose, focal_length=128, resolution=256, num_samplePoints=1000):
	def dense_correspondence_compute_tx_tz_theta(current_pose, next_pose):
		x1, y1, theta1 = next_pose
		x0, y0, theta0 = current_pose
		phi = atan2(y1-y0, x1-x0)
		gamma = minus_theta_fn(theta0, phi)
		dist = math.sqrt((x1-x0)**2 + (y1-y0)**2)
		tz = dist * cos(gamma)
		tx = -dist * sin(gamma)
		theta_change = (theta1 - theta0)
		#print('dist = {}'.format(dist))
		#print('gamma = {}'.format(gamma))
		#print('theta_change = {}'.format(theta_change))
		return tx, tz, theta_change

	kp1 = np.empty((2, num_samplePoints))
	for i in range(num_samplePoints):
		kp1[0, i] = random.randint(0, resolution-1)
		kp1[1, i] = random.randint(0, resolution-1)

	## camera intrinsic matrix
	K = np.array([[focal_length, 0, focal_length], [0, focal_length, focal_length], [0, 0, 1]])
	## expand kp1 from 2 dimensions to 3 dimensions
	kp1_3d = np.ones((3, kp1.shape[1]))
	kp1_3d[:2, :] = kp1

	## backproject kp1_3d through inverse of K and get kp1_3d. x=KX, X is in the camera frame
	## Now kp1_3d still have the third dimension Z to be 1.0. This is the world coordinates in camera frame after projection.
	kp1_3d = LA.inv(K).dot(kp1_3d)

	## backproject kp1_3d into world coords kp1_4d by using gt-depth
	## Now kp1_4d has coords in world frame if camera1 (current) frame coincide with the world frame
	kp1_4d = np.ones((4, kp1.shape[1]))
	for i in range(kp1.shape[1]):
		Z = current_depth[int(kp1[1, i]), int(kp1[0, i])]
		kp1_4d[2, i] = Z
		kp1_4d[0, i] = Z * kp1_3d[0, i]
		kp1_4d[1, i] = Z * kp1_3d[1, i]

	## first compute the rotation and translation from current frame to goal frame
	tx, tz, theta = dense_correspondence_compute_tx_tz_theta(current_pose, goal_pose)
	bad = []
	for i in range(kp1.shape[1]):
		Z = current_depth[int(kp1[1, i]), int(kp1[0, i])]
		if Z <= tz:
			bad.append(i)
	#print('bad = {}'.format(bad))
	R = np.array([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]])
	T = np.array([tx, 0, tz])
	## then compute the transformation matrix from goal frame to current frame
	## thransformation matrix is the camera2's extrinsic matrix
	transformation_matrix = np.empty((3, 4))
	transformation_matrix[:3, :3] = R.T
	transformation_matrix[:3, 3] = -R.T.dot(T)

	## transform kp1_4d from camera1(current) frame to camera2(goal) frame through transformation matrix
	kp2_3d = transformation_matrix.dot(kp1_4d)
	kp2_3d_cpy = kp2_3d.copy()
	## project kp2_3d into a plane. So that the Z dimension is 1.0
	kp2_3d[0, :] = kp2_3d[0, :] / kp2_3d[2, :]
	kp2_3d[1, :] = kp2_3d[1, :] / kp2_3d[2, :]
	kp2_3d[2, :] = kp2_3d[2, :] / kp2_3d[2, :]
	## pass kp2_3d through intrinsic matrix
	kp2 = K.dot(kp2_3d)
	## give up the last dimension of kp2. Only keep the first and second dimension 'u' and 'v'
	kp2 = np.floor(kp2[:2, :])

	good = []
	for i in range(kp1.shape[1]):
		u_prime = kp2[0, i]
		v_prime = kp2[1, i]
		if u_prime < resolution and u_prime >= 0 and v_prime >= 0 and v_prime < resolution:
			good.append(i)
	bad_depth = []
	for i in good:
		goal_Z = goal_depth[int(kp2[1, i]), int(kp2[0, i])]
		if abs(goal_Z - kp2_3d_cpy[2, i]) > 0.1:
			bad_depth.append(i)
	good_remove_bad = []
	for i in good:
		if i not in bad and i not in bad_depth:
				good_remove_bad.append(i)
	kp2 = kp2[::-1, good_remove_bad]
	kp1 = kp1[::-1, good_remove_bad]
	return kp1, kp2

## same as update_current_pose_for_vs() in util_ddpg.py
def update_current_pose (current_pose, vx, vz, omegay):
	x, y, theta = current_pose
	#theta = theta + omegay
	vx_theta = minus_theta_fn(pi/2, theta)
	x = x + vz * cos(theta) + vx * cos(vx_theta)
	y = y + vz * sin(theta) + vx * sin(vx_theta)
	theta = theta + omegay
	goal_pose = [x, y, theta]
	return goal_pose

## compute gt action from start_pose given computed velocities (might be too large)
## return next_pose and real velocities and flag_stop
def goToPose_one_step (start_pose, computed_vx, computed_vz, computed_omegay):
	Kalpha, Kbeta, Krho, delta = 1.0, -0.3, 0.5, 1.0

	x = np.zeros(2, dtype=np.float32)
	y = np.zeros(2, dtype=np.float32)
	theta = np.zeros(2, dtype=np.float32)
	rho = np.zeros(2, dtype=np.float32)
	alpha = np.zeros(2, dtype=np.float32)
	beta  = np.zeros(2, dtype=np.float32)

	## compute goal pose from computed velocities
	goal_pose = update_current_pose (start_pose, computed_vx, computed_vz, computed_omegay)

	## transformation matrix T is from world frame to goal pose frame
	T = np.array([cos(goal_pose[2]), -sin(goal_pose[2]), goal_pose[0], sin(goal_pose[2]), cos(goal_pose[2]), goal_pose[1], 0, 0, 1]).reshape((3,3))
	temp = LA.inv(T).dot(np.array([start_pose[0], start_pose[1], 1]).reshape(3, 1))
	x[0] = temp[0]
	y[0] = temp[1]
	theta[0] = minus_theta_fn(goal_pose[2], start_pose[2])
	x_g, y_g, theta_g = 0.0, 0.0, 0.0

	## move to the next step ===========================================================================
	i = 0

	## (x, y) to (rho, alpha, betta)
	rho[i] = sqrt((x_g - x[i])**2 + (y_g - y[i])**2)
	if rho[i] >= 0.01:
		alpha[i] = minus_theta_fn(theta[i], atan2((y_g - y[i]), (x_g - x[i])))
		beta[i] = (-1) * plus_theta_fn(theta[i], alpha[i])
	else:
		alpha[i] = 0.0
		beta[i] = theta[i]
	#print('rho = {:.2f}, alpha = {:.2f}, beta = {:.2f}, theta[i] = {:.2f}'.format(rho[i], alpha[i], beta[i], theta[i]))
	## stopping criteria
	#if rho[i] < 0.05 and abs(alpha[i]) < pi/36 and abs(beta[i]) < pi/36:
	#if rho[i] < 0.05:
		#return start_pose, 0.0, 0.0, True

	## compute the v and omega
	v = Krho * rho[i]
	if v > 0.1:
		v = 0.1
	## round v to a 2-digit float number
	v = round(v, 2)
	omega = Kalpha * alpha[i] + Kbeta * beta[i]
	if abs(omega) > pi/4:
		if omega > 0:
			omega = pi/4
		else:
			omega = -pi/4
	## round omega to times of 10 degree
	omega = omega // (pi/18) * (pi/18)
	if v < 0.02:# and abs(omega) < pi/19:
		return start_pose, 0.0, 0.0, True
	#print('vx = {:.4f}, vz = {:.4f}, omegay = {:.4f}'.format(computed_vx, computed_vz, computed_omegay))
	#print('v = {:.2f}, omega = {:.2f}'.format(v, omega))

	A = np.zeros((3, 2), dtype=np.float32)
	if alpha[i] >= -pi/2 and alpha[i] <= pi/2:
		A[0, 0] = -cos(alpha[i])
		A[1, 0] = sin(alpha[i])/rho[i]
		A[1, 1] = -1
		A[2, 0] = -sin(alpha[i])/rho[i]
	else:
		A[0, 0] = cos(alpha[i])
		A[1, 0] = -sin(alpha[i])/rho[i]
		A[1, 1] = 1
		A[2, 0] = sin(alpha[i])/rho[i]

	temp_v = np.array([v, omega]).reshape((2,1))
	## transform velocity to polar coordinates
	## B[0] is rho dot, B[1] is alpha dot, B[2] is beta dot
	B = A.dot(temp_v)

	## update rho, alpha, beta through computed rho dot, alpha dot, beta dot
	rho[i+1] = rho[i] + delta * B[0]
	alpha[i+1] = alpha[i] + delta * B[1]
	beta[i+1] = beta[i] + delta * B[2]

	polar_theta = plus_theta_fn(-beta[i+1], pi)
	x[i+1] = x_g + rho[i+1] * cos(polar_theta)
	y[i+1] = y_g + rho[i+1] * sin(polar_theta)
	theta[i+1] = plus_theta_fn(theta[i], omega * delta)

	i += 1
	##======================================================================================
	
	x = x[0:i+1]
	y = y[0:i+1]
	theta = theta[0:i+1]

	temp2 = T.dot(np.stack((x, y, np.ones(i+1))))

	current_x = temp2[0, i]
	current_y = temp2[1, i]
	current_theta = plus_theta_fn(theta[i], goal_pose[2])

	current_pose = [current_x, current_y, current_theta]

	return current_pose, v, omega, False ## flag_stop