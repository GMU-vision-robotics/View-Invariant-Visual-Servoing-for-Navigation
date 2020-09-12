import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

img_path = '/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/Allensville/free.png'
img = cv2.imread(img_path, 0)
name = img_path.split('/')[-1][:-4]


# find and display image contours
h, w = img.shape[0:2]
img2, contours, hierarchy= cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# we just need the outer most contour
outer_contour = contours[0]

'''
#cv2.drawContours(img_contour, [outer_contour], 0, (255), 1)
for i in range(len(contours)):
	img_contour = np.zeros((h, w, 3), np.uint8)
	cv2.drawContours(img_contour, contours, i, (0,255,0), 3)
	#plt.imshow(img_contour)
	#plt.show()
	#plt.savefig(name + '_contour.png', dpi=1000)
	cv2.imwrite(name + '_contour_cv2_{}.png'.format(i), img_contour)
'''

'''
rows, _, _ = outer_contour.shape
for j in range(len(outer_contour)-1):
	plt.plot(outer_contour[j:j+2,0,0], outer_contour[j:j+2,0,1], 'g')
plt.savefig(name + '_outer_contour.png')
'''

'''
# find polygonal approx, convex hull and deficits of convexity
img_poly = np.zeros((h, w), np.uint8)
epsilon = 0.01 * cv2.arcLength(outer_contour, True)
approx = cv2.approxPolyDP(outer_contour, epsilon, True)
cv2.drawContours(img_poly, [approx], 0, (255), 1)
plt.imshow(img_poly)
plt.savefig(name + '_poly.png')
'''

for i in range(len(contours)):
	f = open("{}_points.txt".format(i), "w")
	current_cnt = contours[i]
	for j in range(len(current_cnt)):
		f.write(str(current_cnt[j, 0, 0]) + ' ' + str(current_cnt[j, 0, 1]) + '\n')
	## rewrite the first point in current contour
	f.write(str(current_cnt[0, 0, 0]) + ' ' + str(current_cnt[0, 0, 1]) + '\n')
	f.close()