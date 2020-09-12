import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sin, cos, atan2, pi
import numpy.linalg as LA 
from util_visual_servoing import get_train_test_scenes, get_mapper

def minus_theta_fn (previous_theta, current_theta):
	result = current_theta - previous_theta
	if result < -math.pi:
		result += 2 * math.pi
	if result > math.pi:
		result -= 2 * math.pi
	return result

Train_Scenes, Test_Scenes = get_train_test_scenes()
scene_idx = 6
scene_name = Train_Scenes[6]
mapper_scene2z = get_mapper()

#'''
testing_image_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/testing_image_pairs'
pair_idx = 7

current_img = cv2.imread('{}/pair_{}_left.png'.format(testing_image_folder, pair_idx))[:,:,::-1]
goal_img = cv2.imread('{}/pair_{}_right.png'.format(testing_image_folder, pair_idx))[:,:,::-1]
current_depth = np.load('{}/pair_{}_left_depth.npy'.format(testing_image_folder, pair_idx))
goal_depth = np.load('{}/pair_{}_right_depth.npy'.format(testing_image_folder, pair_idx))

presampled_poses = np.load('{}/traj_poses_{}.npy'.format(testing_image_folder, pair_idx))
current_pose = presampled_poses[0]
goal_pose = presampled_poses[-1]
#'''

'''
base_folder = '/home/reza/Datasets/GibsonEnv'
current_depth = np.load('{}/current_depth.npy'.format(base_folder))
current_pose = np.load('{}/final_pose.npy'.format(base_folder))
goal_pose = np.load('{}/goal_pose.npy'.format(base_folder))
current_img = np.load('{}/final_img.npy'.format(base_folder)).astype('int16')
goal_img = np.load('{}/goal_img.npy'.format(base_folder)).astype('int16')
goal_depth = np.load('{}/goal_depth.npy'.format(base_folder))
'''

def sample_gt_dense_correspondences_frow_goalView_with_large_displacement (current_depth, goal_depth, current_pose, goal_pose, gap=32, focal_length=128, resolution=256, start_pixel=31):
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


kp1, kp2 = sample_gt_dense_correspondences_frow_goalView_with_large_displacement(current_depth, goal_depth, current_pose, goal_pose)
num_matches = kp1.shape[1]




#'''

img_combined = np.concatenate((current_img, goal_img), axis=1)
plt.imshow(img_combined)
plt.plot(kp1[1, :], kp1[0, :], 'ro', alpha=0.2)
plt.plot(kp2[1, :]+256, kp2[0, :], 'ro', alpha=0.2)
for i in range(num_matches):
	plt.plot([kp1[1, :], kp2[1, :]+256], 
		[kp1[0, :], kp2[0, :]], 'ro-', alpha=0.2)
plt.show()
#'''

