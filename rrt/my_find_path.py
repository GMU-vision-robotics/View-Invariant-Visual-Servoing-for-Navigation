import cv2
import os
import rrt
import numpy as np

## generate example short trajectory saved in points.npy
'''
directory = '/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/Allensville'
x0 = -0.591084
y0 = 7.3339
x1 = 5.93709
y1 = -0.421058

path_finder = rrt.PathFinder(directory)
print("Loading...")
path_finder.load()
print("Finding...")
solution, lines = path_finder.find(x0, y0, x1, y1)

# convert lines into points
points = []
for i in range(len(lines)):
	x, y = path_finder.pixel_to_point((lines[i][1], lines[i][0]))
	points.append((x, y))

np.save('points.npy', points)
'''

## generate example long trajectory saved in points.npy
directory = '/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/Allensville'
flag_points = [(-0.3, 7.35), (0.04, 4.98), (-0.1, 3.177), (0, 0.0273), (3.8891, 0.2126), (5.941, -0.0697), \
(8.2248, 0.297), (6.5323, 3.4452), (6.4858, 6.4802), (4, 5.75), (3.9354, 4.0), (3.5065, 7.1719), \
(6.9727, 4.8), (3.9354, 3.4), (-0.3, 7.35)] 

path_finder = rrt.PathFinder(directory)
print("Loading...")
path_finder.load()
print("Finding...")

for j in range(len(flag_points)-1):
	x0 = flag_points[j][0]
	y0 = flag_points[j][1]
	x1 = flag_points[j+1][0]
	y1 = flag_points[j+1][1]
	solution, lines = path_finder.find(x0, y0, x1, y1)

	# convert lines into points
	points = []
	for i in range(len(lines)):
		x, y = path_finder.pixel_to_point((lines[i][1], lines[i][0]))
		points.append((x, y))

	np.save('points/points_{}.npy'.format(j), points)
