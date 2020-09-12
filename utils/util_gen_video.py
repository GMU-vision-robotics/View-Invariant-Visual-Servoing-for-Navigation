import numpy as np
import cv2
import os
import glob

import numpy as np
import cv2

base_file_name = 'obs_Allensville_test_5000'

files = [f for f in glob.glob('../hand_manipulate/' + base_file_name + '/*_rgb.jpg')]

# Define the codec and create VideoWriter object
# 24 is fps
out = cv2.VideoWriter('../hand_manipulate/' + base_file_name + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 24, (256, 256))

for i in range(1, len(files) + 1):
	frame = cv2.imread('../hand_manipulate/' + base_file_name + '/' + str(i) + '_rgb.jpg', 1)

	# write the flipped frame
	out.write(frame)
	print('i = %d'%(i))
		
# Release everything if job is finished
out.release()