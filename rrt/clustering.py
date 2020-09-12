import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans, AgglomerativeClustering
import sys
sys.path.append('..')
from math import cos, sin, pi
import matplotlib.colors as colors
import matplotlib.cm as cmx
import rrt
import cv2

#'''
# read reps and coordinates of each observation
X = np.load('/home/reza/work/SPTM/src/my_code/rep_all_obs_Allensville.npy')
poses = np.load('/home/reza/Datasets/GibsonEnv/my_code/rrt/poses_large.npy')
## each pose has 4 parts: x, y, theta (360 degree)

n_centroid = 64 #10 # 500 #50
metric = 'L2' #learned, L2

## draw observation class as different color to see their distribution
if metric == 'L2':
	model = KMeans(n_clusters = n_centroid, random_state=0).fit(X)
#elif metric == 'learned':
#	model = AgglomerativeClustering(affinity='precomputed', n_clusters=n_centroid, linkage='complete').fit(similarity_all)
directory = '/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/Allensville'
path_finder = rrt.PathFinder(directory)
path_finder.load()

free = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/Allensville/free.png')
rows, cols, _ = free.shape

## draw data as acute trianges showingt the pose of each observation
cmap = plt.cm.jet
cNorm  = colors.Normalize(vmin=0, vmax=n_centroid-1)
scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)

plt.imshow(free)
for idx, pose in enumerate(poses):
	x, y = path_finder.point_to_pixel((pose[0], pose[1]))
	theta = pose[2]
	colorVal = scalarMap.to_rgba(model.labels_[idx])
	plt.arrow(x, y, cos(theta), sin(theta), color=colorVal, \
		overhang=1, head_width=0.1, head_length=0.15, width=0.001)
	#plt.plot(x-min_x, y-min_y, marker=(3,1,theta), markersize=1, color=colorVal)

plt.axis([0, cols, 0, rows])
plt.xticks([])
plt.yticks([])
plt.title('{} clusters using {} distance'.format(n_centroid, metric))
plt.savefig('cluster_2224_{}_arrow_using_{}_distance.jpg'.format(n_centroid, metric), bbox_inches='tight', dpi=(400))
plt.close()

np.save('{}_clusters_{}_distance.npy'.format(n_centroid, metric), model.labels_)
#'''




