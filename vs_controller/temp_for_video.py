import numpy as np 
import cv2
import matplotlib.pyplot as plt

'''
base_folder = '/home/reza/Datasets/GibsonEnv/my_code/vs_controller/visualize_dqn_objects/twentyseventh_try_opticalFlow_newDistMetric_Test/Allensville_16/run_6_0'
save_folder = '/home/reza/Datasets/GibsonEnv/my_code/vs_controller/visualize_dqn_objects/twentyseventh_try_opticalFlow_newDistMetric_Test/Allensville_16/temp_run_6_0'


goal_obj = cv2.imread('{}/{}.jpg'.format(base_folder, 'run_6_object'), 1)


start_img = cv2.imread('{}/{}.jpg'.format(base_folder, 'run_6_start'))
space_img = np.ones((256, 20, 3), dtype=np.uint8)*255
final_img = np.concatenate((goal_obj, space_img, start_img), axis=1)

cv2.imwrite('{}/{}.jpg'.format(save_folder, 'start_img'), final_img)

for i in range(0, 37):s
	current_img = cv2.imread('{}/run_6_step_{}.jpg'.format(base_folder, i))
	final_img = np.concatenate((goal_obj, space_img, current_img), axis=1)
	cv2.imwrite('{}/step_{}.jpg'.format(save_folder, i), final_img)
'''

'''
scene_name_list = ['Allensville', 'Darden', 'Forkland', 'Hiteman', 'Tolstoy', 'Wainscott']
video_idx = 5
scene_name = scene_name_list[video_idx]
point_idx = 4
run_idx = 30
num_frame = 250
base_folder = '{}/{}/point_{}/run_{}'.format('/home/reza/Datasets/GibsonEnv/my_code/vs_controller/for_video', scene_name, point_idx, run_idx)
save_folder = '{}/{}'.format('/home/reza/for video/multienv', video_idx)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#out = cv2.VideoWriter('{}/multiEnv_{}.mp4'.format(save_folder, video_idx), fourcc, 10, (256*2+20, 256))
 
count_img = 0
goal_img = cv2.imread('{}/{}.jpg'.format(base_folder, 'goal_img'), 1)
space_img = np.ones((256, 20, 3), dtype=np.uint8)*255

while count_img <= num_frame:
	if count_img % 5 == 0:
		current_img = cv2.imread('{}/step_{}.jpg'.format(base_folder, count_img), 1)
		final_img = np.concatenate((goal_img, space_img, current_img), axis=1)

		# Write the frame into the file 'output.avi'
		#out.write(final_img)
		cv2.imwrite('{}/step_{}.jpg'.format(save_folder, count_img), final_img)
	count_img += 1


# When everything done, release the video capture and video write objects
#out.release()
'''

# expand the start object image to 256 x 256 image
'''
base_folder = '/home/reza/Datasets/GibsonEnv/my_code/vs_controller/visualize_dqn_objects/twentyseventh_try_opticalFlow_newDistMetric_Test/run_6_0'

goal_obj = cv2.imread('{}/{}.jpg'.format(base_folder, 'run_6_start_object'), 1)
h, w, _ = goal_obj.shape

space_img = np.ones((256, 256, 3), dtype=np.uint8)*255

space_img[110:110+h, 110:110+w, :] = goal_obj
cv2.imwrite('{}/expanded_start_object.jpg'.format(base_folder), space_img)
'''

#'''
run_idx = 6

#base_folder = '/home/reza/Datasets/GibsonEnv/my_code/vs_controller/visualize_dqn_objects/twentyseventh_try_opticalFlow_newDistMetric_Test/Allensville_16/run_6_0'
#base_folder = '/home/reza/Datasets/GibsonEnv/my_code/vs_controller/visualize_dqn_objects/twentyseventh_try_opticalFlow_newDistMetric_Test/Forkland_0/run_1'
base_folder = '/home/reza/Datasets/GibsonEnv/my_code/vs_controller/visualize_dqn_objects/twentyseventh_try_opticalFlow_newDistMetric_Test/run_6_0'

save_folder = '{}/save_folder'.format(base_folder)

start_img = cv2.imread('{}/run_{}_start_copy.jpg'.format(base_folder, run_idx), 1)[:,:,::-1]
goal_obj = cv2.imread('{}/run_{}_expanded_start_object.jpg'.format(base_folder, run_idx), 1)[:,:,::-1]
current_img = cv2.imread('{}/run_{}_start.jpg'.format(base_folder, run_idx), 1)[:,:,::-1]

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,10))
ax[0].imshow(start_img)
ax[0].get_xaxis().set_visible(False)
ax[0].get_yaxis().set_visible(False)
ax[0].set_title("Initial View", fontdict = {'fontsize' : 30})
ax[1].imshow(goal_obj)
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)
ax[1].set_title("Target Object", fontdict = {'fontsize' : 30})
ax[2].imshow(current_img)
ax[2].get_xaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)
ax[2].set_title("Current View", fontdict = {'fontsize' : 30})

fig.tight_layout()
fig.savefig('{}/{}.png'.format(save_folder, 'start_img'))
plt.close()

for i in range(0, 37):
	current_img = cv2.imread('{}/run_{}_step_{}.jpg'.format(base_folder, run_idx, i))[:,:,::-1]

	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,10))
	ax[0].imshow(start_img)
	ax[0].get_xaxis().set_visible(False)
	ax[0].get_yaxis().set_visible(False)
	ax[0].set_title("Initial View", fontdict = {'fontsize' : 30})
	ax[1].imshow(goal_obj)
	ax[1].get_xaxis().set_visible(False)
	ax[1].get_yaxis().set_visible(False)
	ax[1].set_title("Target Object", fontdict = {'fontsize' : 30})
	ax[2].imshow(current_img)
	ax[2].get_xaxis().set_visible(False)
	ax[2].get_yaxis().set_visible(False)
	ax[2].set_title("Current View", fontdict = {'fontsize' : 30})

	fig.tight_layout()
	fig.savefig('{}/step_{}.png'.format(save_folder, i))
	plt.close()
#'''