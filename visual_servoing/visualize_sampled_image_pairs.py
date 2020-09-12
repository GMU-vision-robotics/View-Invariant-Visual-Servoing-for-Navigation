import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing')
from util_visual_servoing import get_train_test_scenes, get_mapper_scene2points
import glob
from math import cos, sin, pi, ceil

Train_Scenes, Test_Scenes = get_train_test_scenes()
mapper_scene2points = get_mapper_scene2points()

def main(scene_idx=0):
	#train_scene_idx = 0
	#scene_name = Test_Scenes[scene_idx]
	#scene_file_addr = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_test/{}'.format(scene_name)
	
	scene_name = Train_Scenes[scene_idx]
	scene_file_addr = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_train/{}'.format(scene_name)
	
	print('scene_name = {}'.format(scene_name))
	num_points = len(mapper_scene2points[scene_name])

	for i in range(num_points):
		print('i = {}'.format(i))
		point_file_addr = '{}/point_{}'.format(scene_file_addr, i)
		right_img_names_all = glob.glob('{}/right*.png'.format(point_file_addr))
		right_img_names = []
		for name in right_img_names_all:
			if 'depth' not in name:
				right_img_names.append(name)
		#assert 1==2
		#right_img_names = right_img_names.sort(key=lambda f: int(filter(str.isdigit, f)))
		right_img_names.sort()

		num_imgs = 1 + len(right_img_names)
		fig = plt.figure(figsize=(22, 18)) #cols, rows
		r, c = ceil(num_imgs*1.0/12), 12

		## left_img
		left_img = cv2.imread('{}/left_img.png'.format(point_file_addr), 1)[:,:,::-1]
		ax = fig.add_subplot(r, c, 1)
		ax.imshow(left_img, shape=(256, 256))
		ax.title.set_text('left_img')
		ax.axis('off')

		for j in range(len(right_img_names)):
			img_name = right_img_names[j]
			right_img = cv2.imread(img_name, 1)[:,:,::-1] 
			ax = fig.add_subplot(r, c, j+2)
			ax.imshow(right_img, shape=(256, 256))
			img_name = img_name[img_name.rfind('/')+1:]
			dist = img_name.split('_')[3]
			theta = img_name.split('_')[5]
			heading = img_name.split('_')[-1][:-4]
			ax.title.set_text('{}-{}-{}'.format(dist, theta, heading))
			ax.axis('off')
		fig.suptitle('point_{} in {}'.format(i, scene_name))
		fig.subplots_adjust(top=0.95)
		fig.savefig('{}/point_{}_visualize.jpg'.format(scene_file_addr, i), bbox_inches='tight', dpi=200)
		plt.close(fig)

if __name__ == "__main__":
	#for i in range(0, len(Test_Scenes)):
	#	main(i)
	#'''
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--scene_idx', type=int, default=0)
	args = parser.parse_args()
	main(args.scene_idx)
	#'''

