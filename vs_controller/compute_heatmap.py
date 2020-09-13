import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import math
import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing')
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/keypointNet')
from util_visual_servoing import sample_gt_random_dense_correspondences
from util_vscontroller import kps2flowMap
from utils_keypointNet import detect_learned_correspondences
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from scipy.ndimage.filters import gaussian_filter

def compute_heatmap (left_img, right_img):
	kp1, kp2 = detect_learned_correspondences(left_img, right_img)
	num_kps = kp1.shape[0]
	## shape of the input image
	nx, ny = left_img.shape[0:2]
	## initialize the heatmap
	blurred = None
	if num_kps >= 3:
		hull = ConvexHull(np.transpose(kp1))
		## build polygon binary map
		poly_verts = kp1[::-1, hull.vertices].T
		# Create vertex coordinates for each grid cell...
		x, y = np.meshgrid(np.arange(nx), np.arange(ny))
		x, y = x.flatten(), y.flatten()
		points = np.vstack((x, y)).T
		path = Path(poly_verts)
		grid = path.contains_points(points)
		grid = grid.reshape((nx,ny)).astype(float)
		## gaussian blur
		blurred = gaussian_filter(grid, sigma=7)
	else:
		grid = np.zeros((ny, nx), dtype=np.float32)
		for i in range(num_kps):
			y, x = kp1[i]
			grid[y, x] = 1.0
		blurred = gaussian_filter(grid, sigma=7)
	return blurred

scene_name
base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_test'
point_image_folder = '{}/{}/point_{}'.format(base_folder, scene_name, point_idx)
point_pose_npy_file = np.load('{}/{}/point_{}_poses.npy'.format(base_folder, scene_name, point_idx))


'''
kp1 = np.array([100, 200, 200, 100])
kp1 = kp1.reshape((2, 2))
num_kps= kp1.shape[0]
grid = np.zeros((256, 256), dtype=np.float32)
for i in range(num_kps):
	y, x = kp1[i]
	grid[y, x] = 1.0
blurred = gaussian_filter(grid, sigma=7)
'''

'''
scene_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_train/Collierville'
point_folder = '{}/{}'.format(scene_folder, 'point_0')

left_img = cv2.imread('{}/{}.png'.format(point_folder, 'left_img'), 1)[:, :, ::-1]
right_img = cv2.imread('{}/{}.png'.format(point_folder, 'right_img_dist_20_theta_15_heading_0'), 1)[:, :, ::-1]

left_depth = np.load('{}/{}_depth.npy'.format(point_folder, 'left_img'))
right_depth = np.load('{}/{}_depth.npy'.format(point_folder, 'right_img_dist_20_theta_15_heading_0'))

pose_file = np.load('{}/{}_poses.npy'.format(scene_folder, 'point_0'))

kp1, kp2 = detect_learned_correspondences(left_img, right_img)

hull = ConvexHull(np.transpose(kp1))


fig = plt.figure(figsize=(10, 15))
r, c = 3, 2
ax = fig.add_subplot(r, c, 1)
ax.imshow(left_img)
ax.title.set_text('start view')
ax.axis('off')
ax = fig.add_subplot(r, c, 2)
ax.imshow(right_img)
ax.title.set_text('target view')
ax.axis('off')

## draw convex hull
ax = fig.add_subplot(r, c, 3)
ax.imshow(left_img)
ax.plot(kp1[1, :], kp1[0, :], 'o')
ax.plot(kp1[1, hull.vertices], kp1[0, hull.vertices], 'k-')
## connect last two points
ax.plot(kp1[1, hull.vertices[[-1, 0]]], kp1[0, hull.vertices[[-1, 0]]], 'k-')
ax.title.set_text('convex hull')
ax.axis('off')

## build polygon binary map
nx, ny = left_img.shape[0:2]
poly_verts = kp1[::-1, hull.vertices].T
# Create vertex coordinates for each grid cell...
x, y = np.meshgrid(np.arange(nx), np.arange(ny))
x, y = x.flatten(), y.flatten()
points = np.vstack((x, y)).T
path = Path(poly_verts)
grid = path.contains_points(points)
grid = grid.reshape((nx,ny)).astype(float)

## gaussian blur
blurred = gaussian_filter(grid, sigma=7)
ax = fig.add_subplot(r, c, 4)
ax.imshow(blurred)
ax.title.set_text('heatmap')
ax.axis('off')
#ax.colorbar()


ax = fig.add_subplot(r, c, 5)
num_matches = kp1.shape[1]
img_combined = np.concatenate((left_img, right_img), axis=1)
ax.imshow(img_combined)
ax.plot(kp1[1, :], kp1[0, :], 'ro', alpha=0.2)
ax.plot(kp2[1, :]+256, kp2[0, :], 'ro', alpha=0.2)
for i in range(num_matches):
	ax.plot([kp1[1, :], kp2[1, :]+256], 
		[kp1[0, :], kp2[0, :]], 'ro-', alpha=0.2)
ax.title.set_text('detected matches')
ax.axis('off')
#plt.show()
fig.savefig('{}.jpg'.format('visualize_heatmap2'), bbox_inches='tight')
'''
