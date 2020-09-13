import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import cv2
import sys
from collections import deque
from math import sin, cos, sqrt, pi, atan2, floor
import random
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop')
from util_visual_servoing import sample_gt_random_dense_correspondences, sample_gt_dense_correspondences, sample_gt_dense_correspondences_in_bbox
from util import plus_theta_fn, minus_theta_fn


## convert input keypoint paris into a flowmap
def kps2flowMap (kp1, kp2, h=256, w=256):
	## initialize flow maps
	optical_flow = np.zeros((h, w, 3), dtype=np.float32)
	mask = np.zeros((h, w, 1), dtype=np.float32)

	num_matches = kp1.shape[1]
	for i in range(num_matches):
		y1, x1 = kp1[:, i]
		y2, x2 = kp2[:, i]
		y_displacement = y2 - y1
		x_displacement = x2 - x1
		y1 = int(y1)
		x1 = int(x1)
		optical_flow[y1, x1, :] = [y_displacement, x_displacement, 0.0]
		mask[y1, x1, 0] = 1.0
	return optical_flow, mask

def genGtDenseCorrespondenseFlowMap (left_depth, right_depth, left_pose, right_pose):
	kp1, kp2 = sample_gt_dense_correspondences(left_depth, right_depth, left_pose, right_pose, gap=1, focal_length=128, resolution=256, start_pixel=1)
	optical_flow, _ = kps2flowMap(kp1, kp2)
	return optical_flow

def genGtDenseCorrespondenseFlowMapAndMask (left_depth, right_depth, left_pose, right_pose):
	kp1, kp2 = sample_gt_dense_correspondences(left_depth, right_depth, left_pose, right_pose, gap=1, focal_length=128, resolution=256, start_pixel=1)
	optical_flow, mask_flow = kps2flowMap(kp1, kp2)
	return optical_flow, mask_flow

def genGtDenseCorrespondenseFlowMapOnObjects (left_depth, right_depth, left_pose, right_pose, object_bbox):
	kp1, kp2 = sample_gt_dense_correspondences_in_bbox(left_depth, right_depth, left_pose, right_pose, object_bbox, gap=1, focal_length=128, resolution=256, start_pixel=1)
	num_matches = kp1.shape[1]
	flag_having_matches = num_matches > 0
	## compute center of obj_bbox
	x1, y1, x2, y2 = object_bbox
	center_bbox_x, center_bbox_y = floor((x1+x2)/2), floor((y1+y2)/2)
	trans_x, trans_y = 128 - center_bbox_x, 128 - center_bbox_y
	## translate kp2 to the center of the image
	kp2[0, :] = kp2[0, :] + trans_y
	kp2[1, :] = kp2[1, :] + trans_x
	## enlarge the object image
	scale_x = 240.0/(x2-x1)
	scale_y = 240.0/(y2-y1)
	if scale_y >= scale_x:
		scale = scale_x
	else:
		scale = scale_y
	kp2 = (kp2 - 128)*scale+128
	optical_flow = kps2flowMap(kp1, kp2)
	return optical_flow, flag_having_matches

constant_goal_mask = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(256):
	for j in range(256):
		constant_goal_mask[i, j] = [i, j, 0]
def genOverlapAreaOnCurrentView(left_depth, right_depth, left_pose, right_pose):
	kp1, kp2 = sample_gt_dense_correspondences(left_depth, right_depth, left_pose, right_pose, gap=1, focal_length=128, resolution=256, start_pixel=1)
	kp1, kp2 = kp1.astype(np.int16), kp2.astype(np.int16)

	left_mask = np.zeros((256, 256, 3), dtype=np.uint8)

	for i in range(kp1.shape[1]):
		y1, x1 = kp1[:, i]
		y2, x2 = kp2[:, i]
		left_mask[y1, x1] = constant_goal_mask[y2, x2]
	return left_mask

constant_left_mask = np.zeros((256, 256, 3), dtype=np.uint8)
for i in range(256):
	for j in range(256):
		constant_left_mask[i, j] = [i, j, 0]
def genOverlapAreaOnGoalView(left_depth, right_depth, left_pose, right_pose):
	kp2, kp1 = sample_gt_dense_correspondences(right_depth, left_depth, right_pose, left_pose, gap=1, focal_length=128, resolution=256, start_pixel=1)
	kp2, kp1 = kp2.astype(np.int16), kp1.astype(np.int16)

	goal_mask = np.zeros((256, 256, 3), dtype=np.uint8)

	for i in range(kp1.shape[1]):
		y1, x1 = kp1[:, i]
		y2, x2 = kp2[:, i]
		goal_mask[y2, x2] = constant_left_mask[y1, x1]
	return goal_mask

def genErrorOverlapAreaOnCurrentView(left_depth, right_depth, left_pose, right_pose):
	kp1, kp2 = sample_gt_dense_correspondences(left_depth, right_depth, left_pose, right_pose, gap=1, focal_length=128, resolution=256, start_pixel=1)
	kp1, kp2 = kp1.astype(np.int16), kp2.astype(np.int16)

	error_start_mask = np.zeros((256, 256, 3), dtype=np.uint8)
	for i in range(256):
		for j in range(256):
			error_start_mask[i, j] = [i, j, 0]
	for i in range(kp1.shape[1]):
		y1, x1 = kp1[:, i]
		error_start_mask[y1, x1] = [0, 0, 0]
	return error_start_mask

def genErrorOverlapAreaOnGoalView(left_depth, right_depth, left_pose, right_pose):
	kp2, kp1 = sample_gt_dense_correspondences(right_depth, left_depth, right_pose, left_pose, gap=1, focal_length=128, resolution=256, start_pixel=1)
	kp2, kp1 = kp2.astype(np.int16), kp1.astype(np.int16)

	error_goal_mask = np.zeros((256, 256, 3), dtype=np.uint8)
	for i in range(256):
		for j in range(256):
			error_goal_mask[i, j] = [i, j, 0]
	for i in range(kp2.shape[1]):
		y2, x2 = kp2[:, i]
		error_goal_mask[y2, x2] = [0, 0, 0]
	return error_goal_mask

def addGaussianNoise(dataMap, mu=0.0, sigma=10.0):
	h, w, channel = dataMap.shape
	for c in range(channel):
		noise = np.random.normal(mu, sigma, (h, w))
		dataMap[:, :, c] += noise
	dataMap[dataMap < -256.0] = -256.0
	dataMap[dataMap > 256.0] = 256.0
	return dataMap

## keep_prob indicates how many points get a correspondence.
def removeCorrespondenceRandomly(dataMap, keep_prob=1.0):
	h, w, _ = dataMap.shape
	s = np.random.uniform(0, 1, (h, w))
	low_bound = 1.0 - keep_prob
	mask = np.zeros((h, w, 1), dtype=np.float32)
	mask[s >= low_bound] = 1.0
	dataMap = dataMap * mask
	return dataMap

## keep_prob indicates how many points get a correspondence.
def removeCorrespondenceRandomly_withSmoothing(dataMap, keep_prob=1.0):
	h, w, _ = dataMap.shape
	s = np.random.uniform(0, 1, (h, w))
	low_bound = 1.0 - keep_prob
	mask = np.zeros((h, w, 1), dtype=np.float32)
	mask[s >= low_bound] = 1.0
	dataMap = dataMap * mask
	dataMap = cv2.blur(dataMap, (5, 5))
	return dataMap

def normalize_opticalFlow(optical_flow):
	assert optical_flow.shape[2] == 2
	for c in range(2):
		channel = optical_flow[:, :, c]
		optical_flow[:, :, c] = (channel - (-256)) / 512
	return optical_flow

def normalize_depth(depth, confident_dist=5.0):
	assert depth.shape[2] == 1
	depth = depth / confident_dist
	depth[depth > 1] = 1.0
	depth = depth * 255.0
	return depth

def buildActionMapper(flag_fewer_actions=False):
	if not flag_fewer_actions:
		action_table = np.zeros((11, 11, 2), dtype=np.float32)
		## x_axis is Omega_Y
		for x_idx, x_axis in enumerate(range(-10, 11, 2)):
			action_table[:, x_idx, 1] = x_axis/10
			## y_axis is V_z
			for y_idx, y_axis in enumerate(range(-10, 11, 2)):
				action_table[y_idx, :, 0] = y_axis/10
		action_table = action_table.reshape((-1, 2))
	else:
		#action_table = np.zeros((3, 7, 2), dtype=np.float32)
		action_table = np.zeros((1, 7, 2), dtype=np.float32)
		## x_axis is Omega_Y
		for x_idx, x_axis in enumerate([-1, -0.7, -0.3, 0, 0.3, 0.7, 1]):
			action_table[:, x_idx, 1] = x_axis
			## y_axis is V_z
			#for y_idx, y_axis in enumerate([-0.4, 0, 0.4]):
			for y_idx, y_axis in enumerate([0.4]):
				action_table[y_idx, :, 0] = y_axis
		action_table = action_table.reshape((-1, 2))
	return action_table

## compute gt sequence of actions from start_pose to goal_pose
## return list of poses and list of actions [vz, omegay]
def gt_goToPose (start_pose, goal_pose, num_steps=50):
	Kalpha, Kbeta, Krho, delta = 1.0, -0.3, 0.5, 1.0

	x = np.zeros(100, dtype=np.float32)
	y = np.zeros(100, dtype=np.float32)
	theta = np.zeros(100, dtype=np.float32)
	rho = np.zeros(100, dtype=np.float32)
	alpha = np.zeros(100, dtype=np.float32)
	beta  = np.zeros(100, dtype=np.float32)
	velocity_list = []

	## transformation matrix T is from world frame to goal pose frame
	T = np.array([cos(goal_pose[2]), -sin(goal_pose[2]), goal_pose[0], sin(goal_pose[2]), cos(goal_pose[2]), goal_pose[1], 0, 0, 1]).reshape((3,3))
	temp = LA.inv(T).dot(np.array([start_pose[0], start_pose[1], 1]).reshape(3, 1))
	x[0] = temp[0]
	y[0] = temp[1]
	theta[0] = minus_theta_fn(goal_pose[2], start_pose[2])
	x_g, y_g, theta_g = 0.0, 0.0, 0.0

	i = 0
	while True:
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
		if rho[i] < 0.05 and abs(alpha[i]) < pi/36 and abs(beta[i]) < pi/36:
			break

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
		velocity_list.append([v, omega])
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
		if i == num_steps:
			break

	x = x[0:i+1]
	y = y[0:i+1]
	theta = theta[0:i+1]

	odometry = []
	temp2 = T.dot(np.stack((x, y, np.ones(i+1))))

	for j in range(i+1):
		current_x = temp2[0, j]
		current_y = temp2[1, j]
		current_theta = plus_theta_fn(theta[j], goal_pose[2])
		odometry.append([current_x, current_y, current_theta])

	return odometry, velocity_list

class ReplayMemory_vs_SL:
	def __init__(self, max_size):
		self.max_size = max_size
		self.buffer = deque(maxlen=max_size)

	def push(self, state1, state2, action):
		experience = (state1, state2, action)
		self.buffer.append(experience)

	def sample(self, batch_size):
		state1_batch = []
		state2_batch = []
		action_batch = []

		batch = random.sample(self.buffer, batch_size)

		for experience in batch:
			state1, state2, action = experience
			state1_batch.append(state1)
			state2_batch.append(state2)
			action_batch.append(action)

		return state1_batch, state2_batch, action_batch

	def __len__(self):
		return len(self.buffer)

class ReplayMemory_vs_rep:
	def __init__(self, max_size):
		self.max_size = max_size
		self.buffer = deque(maxlen=max_size)

	def push(self, left_img, right_img, goal_img, action):
		experience = (left_img, right_img, goal_img, action)
		self.buffer.append(experience)

	def sample(self, batch_size):
		left_batch = []
		right_batch = []
		goal_batch = []
		action_batch = []

		batch = random.sample(self.buffer, batch_size)

		for experience in batch:
			left_img, right_img, goal_img, action = experience
			left_batch.append(left_img)
			right_batch.append(right_img)
			goal_batch.append(goal_img)
			action_batch.append(action)

		return left_batch, right_batch, goal_batch, action_batch

	def __len__(self):
		return len(self.buffer)

class ReplayMemory_vs_dqn:
	def __init__(self, max_size):
		self.max_size = max_size
		self.buffer = deque(maxlen=max_size)

	def push(self, left_img, goal_img, action, reward, next_left_img, next_goal_img, done):
		experience = (left_img, goal_img, action, np.array([reward]), next_left_img, next_goal_img, done)
		self.buffer.append(experience)

	def sample(self, batch_size):
		left_batch = []
		goal_batch = []
		action_batch = []
		reward_batch = []
		next_left_batch = []
		next_goal_batch = []
		done_batch = []

		batch = random.sample(self.buffer, batch_size)

		for experience in batch:
			left_img, goal_img, action, reward, next_left_img, next_goal_img, done = experience
			left_batch.append(left_img)
			goal_batch.append(goal_img)
			action_batch.append(action)
			reward_batch.append(reward)
			next_left_batch.append(next_left_img)
			next_goal_batch.append(next_goal_img)
			done_batch.append(done)

		return left_batch, goal_batch, action_batch, reward_batch, next_left_batch, next_goal_batch, done_batch

	def __len__(self):
		return len(self.buffer)

class ReplayMemory_overlap_dqn:
	def __init__(self, max_size):
		self.max_size = max_size
		self.buffer = deque(maxlen=max_size)

	def push(self, state, action, reward, next_state, done):
		experience = (state, action, np.array([reward]), next_state, done)
		self.buffer.append(experience)

	#'''
	def sample(self, batch_size):
		state_batch = []
		action_batch = []
		reward_batch = []
		next_state_batch = []
		done_batch = []

		batch = random.sample(self.buffer, batch_size)

		for experience in batch:
			state, action, reward, next_state, done = experience
			state_batch.append(state)
			action_batch.append(action)
			reward_batch.append(reward)
			next_state_batch.append(next_state)
			done_batch.append(done)

		return state_batch, action_batch, reward_batch, next_state_batch, done_batch
	#'''

	def __len__(self):
		return len(self.buffer)


class ReplayMemory_overlap_dqn_recurrent:
	def __init__(self, max_size):
		self.max_size = max_size
		#self.buffer = deque(maxlen=max_size)
		self.buffer = []
		self.episodes_length = []

	def push(self, episode, episode_length):
		#self.buffer.append(episode)
		if len(self.buffer) + 1 > self.max_size:
			self.buffer[0:1] = []
			self.episodes_length[0:1] = []

		self.buffer.append(episode)
		self.episodes_length.append(episode_length)

	#'''
	def sample(self, batch_size, time_step):
		'''
		sampled_episodes = random.sample(self.buffer, batch_size)
		batch = []
		for episode in sampled_episodes:
			point = np.random.randint(0, len(episode)+1-time_step)
			batch.append(episode[point:point+time_step])
		'''
		## convert episode_length to numpy
		np_episodes_length = np.array(self.episodes_length)
		## find index of episode_length larget than time_step
		idx_length = np.where(np_episodes_length >= time_step)[0]
		#print('idx_length = {}'.format(idx_length))
		## sample episdoes from ids_length_list
		sampled_ids = list(np.random.choice(idx_length, batch_size))

		batch = []
		for episode_id in sampled_ids:
			episode = self.buffer[episode_id]
			#print('len_episode_in_memory = {}'.format(len(episode)))
			point = np.random.randint(0, len(episode)+1-time_step)
			sub_episode = episode[point:point+time_step]
			batch.append(sub_episode)
		return batch
	#'''

	def __len__(self):
		return len(self.buffer)

class ReplayMemory_overlap_dqn_episodes:
	def __init__(self, max_size):
		self.max_size = max_size
		self.buffer = deque(maxlen=max_size)

	def push(self, episode):
		self.buffer.append(episode)

	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		return batch

	def __len__(self):
		return len(self.buffer)



'''
class ReplayMemory_overlap_dqn_recurrent_efficient:
	def __init__(self, max_size):
		self.max_size = max_size
		self.memory = []
		self.episodes_length = []

	def add_episode(self, episode, episode_length):
		if len(self.memory) + 1 > self.max_size:
			self.memory[0:1] = []
			self.episodes_length[0:1] = []

		self.memory.append(episode)
		self.episodes_length.append(episode_length)

	def sample(self, batch_size, time_step):
		## convert episode_length to numpy
		np_episodes_length = np.array(self.episodes_length)
		## find index of episode_length larget than time_step
		idx_length = np.where(np_episodes_length >= time_step)[0]
		#print('idx_length = {}'.format(idx_length))
		## sample episdoes from ids_length_list
		sampled_ids = list(np.random.choice(idx_length, batch_size))

		batch = []
		for episode_id in sampled_ids:
			episode = self.memory[episode_id]
			#print('len_episode_in_memory = {}'.format(len(episode)))
			point = np.random.randint(0, len(episode)+1-time_step)
			sub_episode = episode[point:point+time_step]
			batch.append(sub_episode)

		batch = np.array(batch)
		return np.reshape(batch, [batch_size, time_step, 5])

	def __len__(self):
		return len(self.memory)
'''