import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import math
from math import sin, cos, atan2
from util_visual_servoing import get_train_test_scenes, get_mapper, detect_correspondences, detect_correspondences_for_fixed_kps, detect_correspondences_and_descriptors

import sys
import os, inspect
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/rrt')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop')
import random
from math import cos, sin, pi

Train_Scenes, Test_Scenes = get_train_test_scenes()
scene_idx = 6
scene_name = Train_Scenes[6]
mapper_scene2z = get_mapper()


def compute_tx_tz_theta(current_pose, goal_pose):
	'''
	x1, y1, theta1 = goal_pose
	x0, y0, theta0 = current_pose
	x_change = x1 - x0
	y_change = y1 - y0
	theta_change = theta1 - theta0
	## look from upper side is different from look from downside
	theta_change = -theta_change
	dist = math.sqrt(x_change**2 + y_change**2)
	print('dist = {}'.format(dist))
	tx = dist * cos(theta0)
	tz = dist * sin(theta0)
	print('tx = {}, tz = {}'.format(tx, tz))
	return tx, tz, -theta_change
	'''
	#'''
	x1, y1, theta1 = goal_pose
	x0, y0, theta0 = current_pose
	x_change = x1 - x0
	y_change = y1 - y0
	theta_change = theta1 - theta0
	## look from upper side is different from look from downside
	## So clockwise is the positive direction
	#theta_change = -theta_change
	dist = math.sqrt(x_change**2 + y_change**2)
	#tx = dist * sin(theta0)
	#tz = dist * cos(theta0)
	theta0_real = atan2(y_change, x_change)
	print('theta0_real = {}'.format(theta0_real))
	print('theta0 = {}'.format(theta0))
	print('dist = {}'.format(dist))
	tx = 0.0
	if abs(theta0_real - theta0) > pi/2:
		tz = -dist
	else:
		tz = dist
	return tx, tz, theta_change
	#'''


def estimate_depth(kp1, kp2, current_pose, goal_pose):
	def compute_tx_tz_theta(current_pose, goal_pose):
		'''
		x_change, y_change, theta_change = goal_pose - current_pose
		## look from upper side is different from look from downside
		#theta_change = -theta_change
		dist = math.sqrt(x_change**2 + y_change**2)
		tx = dist * cos(theta_change)
		tz = dist * sin(theta_change)
		return tx, tz, -theta_change
		'''
		'''
		x1, y1, theta1 = goal_pose
		x0, y0, theta0 = current_pose
		x_change = x1 - x0
		y_change = y1 - y0
		theta_change = theta1 - theta0
		## look from upper side is different from look from downside
		theta_change = -theta_change
		dist = math.sqrt(x_change**2 + y_change**2)
		print('dist = {}'.format(dist))
		tx = dist * cos(theta0)
		tz = dist * sin(theta0)
		print('tx = {}, tz = {}'.format(tx, tz))
		return tx, tz, -theta_change
		'''
		x1, y1, theta1 = goal_pose
		x0, y0, theta0 = current_pose
		x_change = x1 - x0
		y_change = y1 - y0
		theta_change = theta1 - theta0
		## look from upper side is different from look from downside
		## So clockwise is the positive direction
		#theta_change = -theta_change
		dist = math.sqrt(x_change**2 + y_change**2)
		#tx = dist * sin(theta0)
		#tz = dist * cos(theta0)
		theta0_real = atan2(y_change, x_change)
		print('theta0_real = {}'.format(theta0_real))
		print('theta0 = {}'.format(theta0))
		print('dist = {}'.format(dist))
		tx = 0.0
		if abs(theta0_real - theta0) > pi/2:
			tz = -dist
		else:
			tz = dist
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

	tx, tz, theta = compute_tx_tz_theta(current_pose, goal_pose)
	temp_m = np.array([[-cos(theta), -sin(theta)], [sin(theta), -cos(theta)]]).dot(np.array([[tx], [tz]]))
	tx, tz = temp_m

	lambda_focal = 128.0
	u0 = lambda_focal
	v0 = lambda_focal
	Zs = np.ones(num_matches)
	for i in range(num_matches):
		v, u = kp1[:, i]
		v_prime, u_prime = kp2[:, i]
		a = lambda_focal*cos(theta) + (u_prime - u0)*sin(theta)
		b = lambda_focal*sin(theta) - (u_prime-u0)*cos(theta)
		c = lambda_focal*tx - (u_prime-u0)*tz
		d = -(v_prime-v0) * sin(theta)
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
		Zs[i] = x[2]
		#print('b = {}'.format(b))
		#print('Ax = {}'.format(A.dot(x)))
		#Z_prime = -x[0]*sin(theta) + x[2]*cos(theta)+tz
		#print('i = {}, Z = {}, Z_prime = {}'.format(i, x[2], Z_prime))
		#print('depth = {}'.format(current_depth[int(v), int(u)]), goal_depth[int(v), int(u)])
	
	return Zs




testing_image_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/testing_image_pairs'
pair_idx = 1

current_img = cv2.imread('{}/pair_{}_left.png'.format(testing_image_folder, pair_idx))[:,:,::-1]
goal_img = cv2.imread('{}/pair_{}_right.png'.format(testing_image_folder, pair_idx))[:,:,::-1]
current_depth = np.load('{}/pair_{}_left_depth.npy'.format(testing_image_folder, pair_idx))
goal_depth = np.load('{}/pair_{}_right_depth.npy'.format(testing_image_folder, pair_idx))

presampled_poses = np.load('{}/traj_poses_{}.npy'.format(testing_image_folder, pair_idx))
current_pose = presampled_poses[0]
goal_pose = presampled_poses[-1]

base_file_addr = '{}'.format('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/try_depth_estimation')

#'''
## find correspondence using geometric verification
kp1, kp2 = detect_correspondences(current_img, goal_img)
kp1 = kp1[:, :5]
kp2 = kp2[:, :5]
num_matches = kp1.shape[1]

kp1_Zs = estimate_depth(kp1, kp2, current_pose, goal_pose)

for i in range(num_matches):
	v, u = kp1[:, i]
	print('gt_depth = {}, estimated_depth = {}'.format(current_depth[int(v), int(u)], kp1_Zs[i]))
#'''
'''
## find correspondence without geometric verification
kp1, _, des1, _ = detect_correspondences_and_descriptors(current_img, goal_img)
kp2, good = detect_correspondences_for_fixed_kps(des1, goal_img)
num_matches = kp1.shape[1]

kp1_Zs = estimate_depth(kp1, kp2, current_pose, goal_pose)

for i in range(num_matches):
	v, u = kp1[:, i]
	print('gt_depth = {}, estimated_depth = {}'.format(current_depth[int(v), int(u)], kp1_Zs[i]))
'''

kp1_3dim = np.ones((3, num_matches))
kp2_3dim = np.ones((3, num_matches))
kp1_3dim[:2, :] = kp1
kp2_3dim[:2, :] = kp2
K = np.array([[128.0, 0, 128.0], [0, 128.0, 128.0], [0, 0, 1.0]])
X1 = LA.inv(K).dot(kp1_3dim)
X2 = LA.inv(K).dot(kp2_3dim)

tx, tz, theta_change = compute_tx_tz_theta(current_pose, goal_pose)
theta = theta_change
#theta = 0.0
R = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
T = np.array([tx, 0, tz])


R = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
print('tx = {}, tz = {}'.format(tx, tz))
temp_m = -R.T.dot(np.array([[tx], [tz]]))
cam1_tx = tx
cam1_tz = tz
tx, tz = temp_m
print('tx = {}, tz = {}'.format(tx, tz))


transform_matrix = np.zeros((4 ,4))
transform_matrix[0, 0] = cos(theta)
transform_matrix[0, 2] = -sin(theta)
transform_matrix[0, 3] = tx
transform_matrix[1, 1] = 1
transform_matrix[2, 0] = sin(theta)
transform_matrix[2, 2] = cos(theta)
transform_matrix[2, 3] = tz
transform_matrix[3, 3] = 1


#assert 1==2
lambda_focal = 128.0
u0 = lambda_focal
v0 = lambda_focal
intrinsic_matrix = np.zeros((3, 3))
intrinsic_matrix[0, 0] = lambda_focal
intrinsic_matrix[0, 2] = u0
intrinsic_matrix[1, 1] = lambda_focal
intrinsic_matrix[1, 2] = v0
intrinsic_matrix[2, 2] = 1


Zs = np.ones(num_matches)
for i in range(num_matches):
	v, u = kp1[:, i]
	v_prime, u_prime = kp2[:, i]
	a = (u_prime - u0)*sin(theta) - lambda_focal*cos(theta) 
	b = lambda_focal*sin(theta) + (u_prime-u0)*cos(theta)
	c = lambda_focal*tx - (u_prime-u0)*tz
	d = (v_prime-v0) * sin(theta)
	e = (v_prime-v0) * cos(theta)
	f = -(v_prime - v0) * tz
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
	b[2, 0] = c 
	b[3, 0] = f 
	#x = svdsolve(A, b)
	x, _, _, _ = np.linalg.lstsq(A, b)
	Zs[i] = x[2]
	print('b = {}'.format(b))
	print('Ax = {}'.format(A.dot(x)))
	print('X = {}, Y = {}, Z = {}'.format(x[0], x[1], x[2]))
	print('current_depth = {}'.format(current_depth[int(v), int(u)]))
	X = x[0]
	Y = x[1]
	Z = x[2]
	computed_u = lambda_focal * X / Z + u0
	computed_v = lambda_focal * Y / Z + v0
	print('u = {}, computed_u = {}, v = {}, computed_v = {}'.format(u, computed_u, v, computed_v))

	#'''
	point_cam2 = np.array([X, Y, Z, 1])
	point_cam2 = point_cam2.reshape(4, 1)

	point_cam2 = transform_matrix.dot(point_cam2)
	point_cam2 = point_cam2[:3, 0]
	#assert 1==2
	point_cam2 = intrinsic_matrix.dot(point_cam2)
	computed_u_prime = point_cam2[0] / point_cam2[2]
	computed_v_prime = point_cam2[1] / point_cam2[2]
	#'''
	print('u_prime = {}, computed_u_prime = {}, v_prime = {}, computed_v_prime = {}'.format(u_prime, computed_u_prime, v_prime, computed_v_prime))
	#print('X = {}, Y = {}, Z = {}'.format(X, Y, Z))

'''
img_combined = np.concatenate((current_img, goal_img), axis=1)
plt.imshow(img_combined)
plt.plot(kp1[1, :], kp1[0, :], 'ro')
plt.plot(kp2[1, :]+256, kp2[0, :], 'ro')
for i in range(num_matches):
	plt.plot([kp1[1, :], kp2[1, :]+256], 
		[kp1[0, :], kp2[0, :]], 'ro-')
plt.show()
#plt.savefig('{}/step_{}.jpg'.format(matches_file_addr, count_steps), bbox_inches='tight')
#plt.close()
'''