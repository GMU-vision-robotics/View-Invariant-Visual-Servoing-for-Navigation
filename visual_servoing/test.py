import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
#from util_visual_servoing import get_train_test_scenes

target_h = 256
target_w = 256

target_obj_name = 'sofa1'
img = cv2.imread('{}/{}.png'.format('/home/reza/Datasets/GibsonEnv/my_code/visual_servoing/target_objects', target_obj_name), 1)[:,:,::-1]

h, w, _ = img.shape
assert h <= target_h
assert w <= target_w

final_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
left_upper_corner_y = (target_h - h) // 2
left_upper_corner_x = (target_w - w) // 2
final_img[left_upper_corner_y:left_upper_corner_y+h, left_upper_corner_x:left_upper_corner_x+w] = img

plt.imshow(final_img)
plt.show()