import numpy as np
import matplotlib.pyplot as plt
import math

## result is in the range [-pi, pi]
def minus_theta_fn (previous_theta, current_theta):
	result = current_theta - previous_theta
	if result < -math.pi:
		result += 2 * math.pi
	if result > math.pi:
		result -= 2 * math.pi
	return result

def plus_theta_fn (previous_theta, current_theta):
	result = current_theta + previous_theta
	if result < -math.pi:
		result += 2 * math.pi
	if result > math.pi:
		result -= 2 * math.pi
	return result

## threshold for forward and turning
thresh_forward = 0.01
upper_thresh_theta = math.pi / 6
lower_thresh_theta = math.pi / 12

# pose: (x, y, theta)
poses = []
actions = []

previous_theta = 0
for f in range(14):
	points = np.load('points/points_{}.npy'.format(f))
	## compute theta for each point except the last one
	## theta is in the range [-pi, pi]
	thetas = []
	for i in range(len(points) - 1):
		p1 = points[i]
		p2 = points[i + 1]
		current_theta = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
		thetas.append(current_theta)

	for i in range(len(points) - 1):
		p1 = points[i]
		p2 = points[i+1]

		current_theta = thetas[i]
		## so that previous_theta is same as current_theta for the first point
		if i == 0 and f == 0:
			previous_theta = current_theta
		## first point is not the result of an action
		## append an action before introduce a new pose
		if i != 0:
			## forward: 0, left: 3, right 2
			actions.append(0)
		## after turning, previous theta is changed into current_theta
		if i != 0 or f == 0:
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
		## didn't add forward points to reduce redundancy

		## we don't need to add p2 to poses unless p2 is the last point in points
		if i + 1 == len(points) - 1:
			actions.append(0)
			pose = (p2[0], p2[1], current_theta)
			poses.append(pose)

np.save('poses_large.npy', poses)
np.save('poses_actions_large.npy', actions)	
	
