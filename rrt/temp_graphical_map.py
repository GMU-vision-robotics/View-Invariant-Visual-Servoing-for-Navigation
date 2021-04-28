import cv2 
import numpy as np
from matplotlib import pyplot as plt
from DFS import graph, diffDriveT
import my_code.rrt.rrt as rrt
import math
from my_code.CVPR_workshop.util import minus_theta_fn, plus_theta_fn

traj_id = 2

scene = 'Wainscott'
scene_folder = '/home/yimeng/Datasets/GibsonEnv_old/gibson/assets/dataset/{}_for_rrt'.format(scene)

directory = '/home/yimeng/Datasets/GibsonEnv_old/gibson/assets/dataset/{}_for_rrt'.format(scene)
path_finder = rrt.PathFinder(directory)
print("Loading...")
path_finder.load()
print("Finding...")

upper_thresh_theta = math.pi / 6
lower_thresh_theta = math.pi / 12

#================================ load free map and the graphical map ==================================
free_map = cv2.imread('{}/free.png'.format(scene_folder), 0)
topo_map = cv2.imread('{}/{}_output.png'.format(scene_folder, scene), 0)
graph_npy = np.load('{}/v_and_e.npy'.format(scene_folder), allow_pickle=True).item()

plt.imshow(topo_map)
plt.show()

H_topo, W_topo = topo_map.shape
H_free, W_free = free_map.shape

start_node = (int(672/W_topo*W_free), int(1154/H_topo*H_free))
end_node = (int(571/W_topo*W_free), int(1482/H_topo*H_free))
print('start_node = {}, end_node = {}'.format(start_node, end_node))

#================================ Run BFS to generate the path ===========================================
G = graph(graph_npy)
paths = G.find_paths(start_node, end_node)

path = paths[0]

fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
rx_up = [x for (x, y) in path]
ry_up = [y for (x, y) in path]
ax.imshow(free_map, cmap='gray')
ax.scatter(x=rx_up, y=ry_up, c='r', s=1) 
for i in range(1, len(rx_up)):
	ax.plot([rx_up[i-1], rx_up[i]], [ry_up[i-1], ry_up[i]], 'b-', lw=1)
ax.plot(rx_up[0], ry_up[0], c='y', marker='*')
#plt.show()
fig.savefig('my_code/rrt/topo_path_obs/{}_traj_{}_topo.png'.format(scene, traj_id), format='png', dpi=500, bbox_inches='tight', pad_inches=0)

dict_poses = {}
for _ in range(1):
	dict_poses[0] = []

	# visual the last set of waypoints in high-res image	  
	rx_up = [x for (x, y) in path]
	ry_up = [y for (x, y) in path]

	# set linear velocity and follow the set of waypoints
	v = 500
	x0 = rx_up[0]
	y0 = -ry_up[0]
	for i in range(1, len(path)):

		xg = rx_up[i]
		yg = -ry_up[i]

		x, y = path_finder.pixel_to_point((x0, -y0))
		
		if i > 1:
			# rotate the agent to the next agent
			previous_theta = -th0
			current_theta = -math.atan2((yg - y0), (xg -x0))
			## decide turn left or turn right, Flase = left, True = Right
			bool_turn = False
			minus_cur_pre_theta = minus_theta_fn(previous_theta, current_theta)
			if minus_cur_pre_theta < 0:
				bool_turn = True
			## need to turn more than once, since each turn is 30 degree
			while abs(minus_theta_fn(previous_theta, current_theta)) > upper_thresh_theta:
				if bool_turn:
					previous_theta = minus_theta_fn(upper_thresh_theta, previous_theta)
				else:
					previous_theta = plus_theta_fn(upper_thresh_theta, previous_theta)
				dict_poses[0].append((x, y, previous_theta))

		th0 = math.atan2((yg - y0), (xg -x0))
		s0 = [x0, y0, th0]
		dict_poses[0].append((x, y, -th0))	

		err_dist = 50
		err = math.sqrt((xg - x0) ** 2 + (yg - y0) ** 2)
		while err > err_dist:
			w = math.atan2(yg -y0, xg -x0) - th0
			xn, yn, thn = diffDriveT(s0, v, w , 0.1, 1)
			err = math.sqrt((xg - x0)**2 + (yg - y0)**2)
			print('err', err, xn[1], yn[1], thn[1])

			x, y = path_finder.pixel_to_point((xn[1], -yn[1]))

			dict_poses[0].append((x, y, -thn[1]))

			s0 = [xn[1], yn[1], thn[1]]
			x0 = xn[1]
			y0 = yn[1]
			th0 = thn[1]


np.save('my_code/rrt/topo_path_obs/{}_traj_{}_topo_poses.npy'.format(scene, traj_id), dict_poses)

