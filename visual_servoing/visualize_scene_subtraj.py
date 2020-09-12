import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from util_visual_servoing import get_train_test_scenes
from math import cos, sin, pi

cmaps = ['b','g','r','c','m','y','k','w']

temp_I = cv2.imread('/home/reza/Datasets/GibsonEnv/Img1.png')

subtraj_len = 10

Train_Scenes, Test_Scenes = get_train_test_scenes()

## load the scene images, poses and actions
scene_idx = 10
scene_name = Train_Scenes[6]

sys.path.append('/home/reza/Datasets/GibsonEnv/my_code/rrt')
import rrt
rrt_directory = '/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt'.format(scene_name)
path_finder = rrt.PathFinder(rrt_directory)
path_finder.load()
num_nodes = len(path_finder.nodes_x)
free = cv2.imread('/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/{}_for_rrt/free.png'.format(scene_name))

sampled_trajs_npy_folder = '/home/reza/Datasets/GibsonEnv/my_code/CVPR_workshop/sampled_trajs_npy'
sampled_trajs_npy = np.load('{}/{}_trajs.npy'.format(sampled_trajs_npy_folder, scene_name)).item()

subtraj_file = np.load('{}/train_scene_overlapped_subtraj.npy'.format(sampled_trajs_npy_folder)).item()
list_subtraj = subtraj_file[scene_name]

len_list_subtraj = len(list_subtraj)

for i in range(0, 80, 8):

	traj_I = []
	traj_poses = []
	traj_actions = []

	count_qualified = 0
	for j in range(8):
		idx_list_sub_traj = i + j
		traj_idx, subtraj_idx, overlap = list_subtraj[idx_list_sub_traj]

		if overlap < 3:
			## load presampled images, poses and actions
			presampled_start_idx = subtraj_len * subtraj_idx
			presampled_end_idx = subtraj_len * (subtraj_idx + 1)
			presampled_I = sampled_trajs_npy[traj_idx]['images'][presampled_start_idx:presampled_end_idx]
			presampled_actions = sampled_trajs_npy[traj_idx]['actions'][presampled_start_idx:presampled_end_idx]
			presampled_poses = sampled_trajs_npy[traj_idx]['poses'][presampled_start_idx:presampled_end_idx]

			count_qualified += 1
			traj_I.append(presampled_I)
			traj_poses.append(presampled_poses)
			traj_actions.append(presampled_actions)

			print('count_qualified = {}, traj_idx = {}, subtraj_idx = {}, overlap = {}'.format(count_qualified, 
				traj_idx, subtraj_idx, overlap))


	file_addr = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/visualize_subtrajs'
	img_name = '{}_pose_result_{}_overlap_{}.jpg'.format(scene_name, i, overlap)
	print('img_name = {}'.format(img_name))
	
	## plot the poses
	rows, cols, _ = free.shape
	plt.imshow(free)
	for n in range(count_qualified):
		subtraj_poses = traj_poses[n]
		for m in range(subtraj_len):
			pose = subtraj_poses[m]
			x, y = path_finder.point_to_pixel((pose[0], pose[1]))
			theta = pose[2]
			plt.arrow(x, y, cos(theta), sin(theta), color=cmaps[n], \
				overhang=1, head_width=0.1, head_length=0.15, width=0.001)

	plt.axis([0, cols, 0, rows])
	plt.xticks([])
	plt.yticks([])
	plt.savefig('{}/{}'.format(file_addr, img_name), bbox_inches='tight', dpi=(400))
	plt.close()

	## plot the images
	action_map = {0:'forward', 1:'stop', 2:'turn right', 3:'turn left'}
	fig = plt.figure(figsize=(10, 9)) #cols, rows
	r, c = 9, 10

	for n in range(count_qualified):
		subtraj_actions = traj_actions[n]
		subtraj_I = traj_I[n]
		for m in range(subtraj_len):
			img = subtraj_I[m]
			ax = fig.add_subplot(r, c, n*subtraj_len+1+m)
			ax.imshow(img, shape=(256, 256))
			ax.title.set_text('{}'.format(action_map[subtraj_actions[m]]))
			ax.axis('off')
	
	fig.savefig('{}/{}_visualize_obs_{}_overlap_{}.jpg'.format(file_addr, scene_name, 
		i, overlap), bbox_inches='tight')
	plt.close(fig)