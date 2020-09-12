import cv2
import os
import rrt
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi

directory = '/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/Allensville'
path_finder = rrt.PathFinder(directory)
path_finder.load()

free = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/Allensville/free.png')
rows, cols, _ = free.shape

poses = np.load('poses_large.npy')

plt.imshow(free)

#'''
## draw flag points
flag_points = [(-0.3, 7.35), (0.04, 4.98), (-0.1, 3.177), (0, 0.0273), (3.8891, 0.2126), (5.941, -0.0697), \
(8.2248, 0.297), (6.5323, 3.4452), (6.4858, 6.4802), (4, 5.75), (3.9354, 4.0), (3.5065, 7.1719), \
(6.9727, 4.8), (3.9354, 3.4), (-0.3, 7.35)]  

x_list = []
y_list = []
for j in range(len(flag_points)):
	x, y = path_finder.point_to_pixel((flag_points[j][0], flag_points[j][1]))
	x_list.append(x)
	y_list.append(y)
plt.plot(x_list, y_list, 'ro')
#'''
'''
for i in range(len(poses)):
	pose = poses[i]
	x, y = path_finder.point_to_pixel((pose[0], pose[1]))
	theta = pose[2]
	plt.arrow(x, y, cos(theta), sin(theta), color='b', \
		overhang=1, head_width=0.1, head_length=0.15, width=0.001)
'''

plt.axis([0, cols, 0, rows])
plt.xticks([])
plt.yticks([])
#plt.savefig('visualize_poses_large.jpg', bbox_inches='tight', dpi=(400))
plt.savefig('visualize_flag_points.jpg', bbox_inches='tight', dpi=(400))
plt.close()
#plt.show()