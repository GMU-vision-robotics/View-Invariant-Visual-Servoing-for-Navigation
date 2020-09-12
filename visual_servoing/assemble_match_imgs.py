import cv2
import matplotlib.pyplot as plt 
import numpy as np

file_addr = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/test_vs'
pair_idx = 4
actions = np.load('{}/actions_pair_{}.npy'.format(file_addr, pair_idx))

len_actions = len(actions)

match_img_addr = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/test_vs/{}'.format(pair_idx)
action_list = [0,2,5,6,9]
for i in range(len_actions):
	img0 = cv2.imread('{}/step_{}_action_{}.jpg'.format(match_img_addr, i, 0))[:,:,::-1]
	img2 = cv2.imread('{}/step_{}_action_{}.jpg'.format(match_img_addr, i, 2))[:,:,::-1]
	img5 = cv2.imread('{}/step_{}_action_{}.jpg'.format(match_img_addr, i, 5))[:,:,::-1]
	img6 = cv2.imread('{}/step_{}_action_{}.jpg'.format(match_img_addr, i, 6))[:,:,::-1]
	img9 = cv2.imread('{}/step_{}_action_{}.jpg'.format(match_img_addr, i, 9))[:,:,::-1]
	imgs = []
	imgs.append(img0)
	imgs.append(img2)
	imgs.append(img5)
	imgs.append(img6)
	imgs.append(img9)

	action_map = {0:'forward', 1:'turn right 30 degree', 2:'turn right 15 degree', 3:'turn left 30 degree', 
		4:'turn left 15 degree'}

	fig = plt.figure(figsize=(100, 15))
	#fig.subplots_adjust(hspace=0.3, wspace=None)
	r, c = 5, 1
	for j in range(len(imgs)):
		img = imgs[j]
		ax = fig.add_subplot(r, c, j+1)
		ax.imshow(img)
		ax.title.set_text('{}'.format(action_map[j]))
		ax.axis('off')
	fig.suptitle('step {}, action = {}'.format(i, actions[i]))
	fig.savefig('{}/pair_{}_step_{}.jpg'.format(file_addr, pair_idx, i), bbox_inches='tight')
	plt.close(fig)