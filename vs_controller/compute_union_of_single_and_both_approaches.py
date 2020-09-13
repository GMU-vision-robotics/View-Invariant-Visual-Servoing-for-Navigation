import numpy as np

f1 = open('successful_runs_eighteenth_try_siamese.txt', 'r')
f2 = open('successful_runs_nineteenth_try_corresMapCurrentView_goToPose_metric.txt', 'r')

list1 = []
list2 = []

for i in range(85):
	list1.append(f1.readline()[:-1])

for i in range(98):
	list2.append(f2.readline()[:-1])

## find the union of the two lists
final_result = list1
for name2 in list2:
	flag = False
	for name in list1:
		if name2 == name:
			flag = True
	if not flag:
		final_result.append(name2)

## find the offset of the two lists
point_pose_npy_file = np.load('{}/{}/point_{}_poses.npy'.format('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/sample_image_pairs_test', 'Allensville', 0))

offset = []
offset_id = []
for right_img_idx in range(1, len(point_pose_npy_file)):
	right_img_name = point_pose_npy_file[right_img_idx]['img_name']
	flag = False
	for name in final_result:
		if right_img_name == name:
			flag = True
	if not flag:
		offset.append(right_img_name)
		offset_id.append(right_img_idx)




