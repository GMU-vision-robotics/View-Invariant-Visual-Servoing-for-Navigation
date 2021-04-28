import cv2 
import numpy as np
from matplotlib import pyplot as plt
import my_rrt

# load the occupancy img
free = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/Allensville/free.png', 0)

#==============================build the rrt tree to cover the whole environment===================================
edges_from_px, edges_to_px, nodes_x, nodes_y, edges = my_rrt.make_rrt(free)

floor_map = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/Allensville/free.png', 1)
for i, edge_from_px in enumerate(edges_from_px):
	cv2.line(floor_map, edge_from_px, edges_to_px[i], (255, 0, 0), thickness=5)

cv2.imwrite('floor_map.png', floor_map)
#plt.imshow(floor_map)
#plt.show()

path_finder = my_rrt.PathFinder(nodes_x, nodes_y, edges, free)

#=================================== find a path from pixel (1000, 1000) to pixel (4000, 4000)======================
start_pixel = (1000, 1000)
target_pixel = (4000, 4000)
_, lines = path_finder.find_path_between_pixels(start_pixel, target_pixel)

traj = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/Allensville/free.png', 1)
for i in range(len(lines)-1):
	y0, x0 = lines[i]
	y1, x1 = lines[i+1]
	cv2.line(traj, (x0, y0), (x1, y1), (255, 0, 0), thickness=10)

cv2.imwrite('traj.png', traj)
#plt.imshow(traj)
#plt.show()





        

            