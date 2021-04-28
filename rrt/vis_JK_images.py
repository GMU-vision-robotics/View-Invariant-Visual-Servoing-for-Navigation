import numpy as np 
import matplotlib.pyplot as plt 
import cv2

image_folder = 'my_code/rrt/JK_path_aug'
path_imgs = [37, 37, 36, 35, 37]


fig, ax = plt.subplots(nrows=len(path_imgs), ncols=min(path_imgs), figsize=(175, 25))
for i in range(len(path_imgs)):
	for j in range(min(path_imgs)):
		img = cv2.imread('{}/path_{}_step_{}.png'.format(image_folder, i, j))[:,:,::-1]
		ax[i][j].imshow(img)
		ax[i][j].get_xaxis().set_visible(False)
		ax[i][j].get_yaxis().set_visible(False)


fig.tight_layout()
#plt.show()
fig.savefig('all_JK_path_aug.jpg')
plt.close()