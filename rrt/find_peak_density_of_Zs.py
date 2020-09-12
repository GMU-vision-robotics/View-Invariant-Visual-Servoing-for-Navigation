import numpy as np 
import json
import matplotlib.pyplot as plt 
from scipy.stats import gaussian_kde

def plot_at_y(arr, val, **kwargs):
	plt.plot(arr, np.zeros_like(arr) + val, 'x', **kwargs)
	plt.show()

#'Allensville', 'Beechwood', 'Coffeen', 'Darden', 'Collierville', 'Corozaj' 'Cosmos', 'Forkland'
#'Hanson', 'Hiteman', 'Klickitat', 'Lakeville', 'Leonardo', 'Lindenwood', 'Markleeville', 'Marstons'
#'McDade', 'Merom', 'Mifflinburg', 'Muleshoe', 'Newfields', 'Noxapater', 'Onaga', 'Pinesdale', 'Pomaria'
#'Ranchester', 'Shelbyville', 'Stockman', 'Tolstoy', 'Uvalda', 'Wainscott', 'Wiconisco', 'Woodbine'
scene_names = [ 'Woodbine']
base_folder = '/home/reza/Datasets/GibsonEnv/gibson/navigation_scenarios/waypoints/tiny/'

for scene_name in scene_names:
	json_file = json.load(open(base_folder + scene_name + '.json'))

	z_list = []
	for i in range(len(json_file)):
		waypoints_list = json_file[i]['waypoints']
		for waypoint in waypoints_list:
			z_list.append(waypoint[2])

	## visualize the z values
	plot_at_y(np.array(z_list), 0)

	density = gaussian_kde(z_list)
	#plot_density(z_list, density(z_list))
	max_density_idx = np.argmax(density(z_list))
	print('{} : z with maximum density = {}'.format(scene_name, z_list[max_density_idx]))