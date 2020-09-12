import numpy as np 
import matplotlib.pyplot as plt

base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/test_sample_image_pairs/largeDisplacement_gtCorrespondence_gtDepth_Vx/rich_visual_cue/scene_Forkland_point_1'
#base_folder = '/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/test_sample_image_pairs/sift_gtDepth/rich_visual_cue/scene_Forkland_point_1'

stat_file = np.load('{}/runs_statistics.npy'.format(base_folder))

## look for unsuccessful test case
print('unsuccessful: ')
count = 0
avg_steps = 0.0
for current_dict in stat_file:
	steps = current_dict['steps']
	avg_steps += steps
	flag = current_dict['success_flag']
	if flag != 'Success':
		count += 1
		print('count = {}, name = {}'.format(count, current_dict['img_name']))
avg_steps = avg_steps * 1.0 / len(stat_file)
print('avg_steps = {}'.format(avg_steps))

'''
## look for test case with no common space
print('No common space: ')
count = 0
for current_dict in stat_file:
	num_matches = current_dict['num_matches']
	if num_matches <= 10:
		count += 1
		print('count = {}, name = {}'.format(count, current_dict['img_name']))
print('----------------------------------------------------------------------------')

## look for test case with obstacles
print('obstacles: ')
count = 0
for current_dict in stat_file:
	steps = current_dict['steps']
	flag = current_dict['success_flag']
	if steps < 50 and flag != 'Success':
		count += 1
		print('count = {}, name = {}'.format(count, current_dict['img_name']))
print('----------------------------------------------------------------------------')

## look similar but failed
print('look similar but failed: ')
count = 0
for current_dict in stat_file:
	num_matches = current_dict['num_matches']
	flag = current_dict['success_flag']
	if num_matches > 40 and flag != 'Success':
		count += 1
		print('count = {}, name = {}'.format(count, current_dict['img_name']))
print('----------------------------------------------------------------------------')

avg_displacement = 0.0
avg_steps = 0.0
count = 0
for current_dict in stat_file:
	num_matches = current_dict['num_matches']
	displacement = current_dict['pixel_displacement']
	steps = current_dict['steps']
	if num_matches > 10:
		avg_displacement += displacement
		avg_steps += steps
		count += 1
avg_displacement = avg_displacement *1.0 / count
avg_steps = avg_steps * 1.0 / count
print('avg_displacement = {}, avg_steps = {}'.format(avg_displacement, avg_steps))
'''


'''
print('analysis: ')
list_success = []
list_fail = []
list_fail_similar = []
for current_dict in stat_file:
	num_matches = current_dict['num_matches']
	flag = current_dict['success_flag']
	theta = current_dict['theta']
	dist = current_dict['dist']
	if num_matches > 10:
		tup = (dist, theta)
		if flag != 'Success':
			if num_matches >= 40:
				list_fail_similar.append(tup)
			else:
				list_fail.append(tup)
		else:
			list_success.append(tup)

## plot
## dist is x, theta is y
success_x = []
success_y = []
fail_x = []
fail_y = []
similar_x = []
similar_y = []
for i in range(len(list_success)):
	success_x.append(list_success[i][0])
	success_y.append(list_success[i][1])
for i in range(len(list_fail)):
	fail_x.append(list_fail[i][0])
	fail_y.append(list_fail[i][1])
for i in range(len(list_fail_similar)):
	similar_x.append(list_fail_similar[i][0])
	similar_y.append(list_fail_similar[i][1])

plt.plot(success_x, success_y, 'ro', label='success')
plt.plot(fail_x, fail_y, 'bo', label='failure')
plt.plot(similar_x, similar_y, 'yo', label='fail but similar')
plt.legend(loc='upper right')
plt.xlabel('L2 distance in meters')
plt.ylabel('heading difference in degree')
plt.grid(True)
plt.title('L2 distance and heading difference of test cases')
plt.show()
'''