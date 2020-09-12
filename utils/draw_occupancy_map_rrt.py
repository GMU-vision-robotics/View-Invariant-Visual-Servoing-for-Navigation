import sys
import matplotlib.pyplot as plt
import meshcut
import numpy as np

Train_Scenes = ['Allensville', 'Uvalda', 'Darden', 'Collierville', 'Cosmos', 'Forkland', 'Hanson', 'Hiteman', 
		'Klickitat', 'Lakeville', 'McDade', 'Mifflinburg', 'Muleshoe', 'Newfields', 'Tolstoy', 'Wainscott']
Test_Scenes = ['Noxapater', 'Onaga', 'Pinesdale', 'Pomaria', 'Shelbyville', 'Stockman', 'Wiconisco']

mapper_scene2z = {}
mapper_scene2z['Allensville'] = 0.47
mapper_scene2z['Uvalda'] = 0.681
mapper_scene2z['Darden'] = 3.40
mapper_scene2z['Collierville'] = 0.46
mapper_scene2z['Cosmos'] = 0.65
mapper_scene2z['Forkland'] = 0.35
mapper_scene2z['Hanson'] = 0.35
mapper_scene2z['Hiteman'] = 0.40
mapper_scene2z['Klickitat'] = 0.40
mapper_scene2z['Lakeville'] = 0.50
mapper_scene2z['Tolstoy'] = 3.50
mapper_scene2z['Wainscott'] = 1.0
mapper_scene2z['McDade'] = 3.16
mapper_scene2z['Mifflinburg'] = 0.417
mapper_scene2z['Muleshoe'] = 1.9
mapper_scene2z['Newfields'] = 0.492
mapper_scene2z['Noxapater'] = 0.523
mapper_scene2z['Onaga'] = 0.46
mapper_scene2z['Pinesdale'] = 0.406
mapper_scene2z['Pomaria'] = 0.492
mapper_scene2z['Shelbyville'] = 0.691
mapper_scene2z['Stockman'] = 3.30
mapper_scene2z['Wiconisco'] = 0.517

## left_x, bottom_y, right_x, top_y
mapper_scene2xyticks = {}
mapper_scene2xyticks['Allensville'] = [-1.5, -1.5, 10.0, 9.0]
mapper_scene2xyticks['Uvalda'] = [-8.0, -7.5, 6.5, 9.5]
mapper_scene2xyticks['Darden'] = [-15.0, -4.0, 2.5, 5.0]
mapper_scene2xyticks['Collierville'] = [-4.0, -2.5, 3.5, 7.5]
mapper_scene2xyticks['Cosmos'] = [-4.0, -3.0, 6.0, 16.0]
mapper_scene2xyticks['Forkland'] = [-10.0, -4.0, 4.5, 5.0]
mapper_scene2xyticks['Hanson'] = [-6.0, -7.0, 2.0, 14.0]
mapper_scene2xyticks['Hiteman'] = [-2.0, -1.5, 14.0, 6.0]
mapper_scene2xyticks['Klickitat'] = [-17.0, -13.0, 8.0, 8.0]
mapper_scene2xyticks['Lakeville'] = [-18.5, -14.0, 2.0, 6.5]
mapper_scene2xyticks['Tolstoy'] = [-11.5, -1.5, 3.5, 16.0]
mapper_scene2xyticks['Wainscott'] = [-6.0, -7.5, 8.5, 15.0]
mapper_scene2xyticks['McDade'] = [-10.0, -17.0, 4.0, 5.0]
mapper_scene2xyticks['Mifflinburg'] = [-5.0, -2.0, 3.0, 9.5]
mapper_scene2xyticks['Muleshoe'] = [-25.0, -13.0, 2.5, 6.5]
mapper_scene2xyticks['Newfields'] = [-12.0, -2.5, 5.0, 10.0]

mapper_scene2xyticks['Noxapater'] = [-4.0, -5.5, 8.5, 4.0]
mapper_scene2xyticks['Onaga'] = [-5.5, -3.0, 2.0, 10.5]
mapper_scene2xyticks['Pinesdale'] = [-8.0, -2.5, 4.5, 22.0]
mapper_scene2xyticks['Pomaria'] = [-14.0, -5.5, 2.5, 7.5]
mapper_scene2xyticks['Shelbyville'] = [-6.0, -5.0, 8.0, 11.5]
mapper_scene2xyticks['Stockman'] = [-7.0, -11.5, 4.0, 4.0]
mapper_scene2xyticks['Wiconisco'] = [-7.0, -3.0, 8.0, 12.5]

def load_obj(fn):
	verts = []
	faces = []
	with open(fn) as f:
		for line in f:
			if line[:2] == 'v ':
				verts.append(list(map(float, line.strip().split()[1:4])))
			if line[:2] == 'f ':
				face = [int(item.split('/')[0]) for item in line.strip().split()[-3:]]
				faces.append(face)
	verts = np.array(verts)
	faces = np.array(faces) - 1
	return verts, faces

def main(scene_idx):
	meshFilePath = 'gibson/assets/dataset/{}_for_rrt/mesh_z_up.obj'.format(Test_Scenes[scene_idx])
	verts, faces = load_obj(meshFilePath)
	z =  mapper_scene2z[Test_Scenes[scene_idx]] # 0.5 is the height of husky, actually it should be 0.37
	# cut the mesh with a surface whose value on z-axis is plane_orig, and its normal is plane_normal vector
	#print('verts: {}'.format(verts[0]))
	print('meshcut ...')
	cross_section = meshcut.cross_section(verts, faces, plane_orig=(0, 0, z), plane_normal=(0, 0, 1))
	plt.figure()
	for item in cross_section:
		for i in range(len(item) - 1):
			#print('item:{}', item[i, 0])
			plt.plot(item[i:i+2,0],item[i:i+2,1], 'b')
			#print('item:{}'.format(item[i:i+2,0]))
	left_x, bottom_y, right_x, top_y = mapper_scene2xyticks[Test_Scenes[scene_idx]]
	x_major_ticks = np.arange(left_x, right_x, 0.5)
	x_minor_ticks = np.arange(left_x, right_x, 0.5)
	y_major_ticks = np.arange(bottom_y, top_y, 0.5)
	y_minor_ticks = np.arange(bottom_y, top_y, 0.5)
	plt.xticks(x_major_ticks)
	#plt.xticks(x_minor_ticks, minor=True)
	plt.yticks(y_major_ticks)
	#plt.yticks(y_minor_ticks, minor=True)
	plt.grid(True)
	## save image to the path
	file_addr = '/home/reza/Datasets/GibsonEnv/my_code/rrt/scene_grid_map'
	img_name = '{}_grid.png'.format(Test_Scenes[scene_idx])
	plt.savefig('{}/{}'.format(file_addr, img_name), bbox_inches='tight', dpi=(400))
	plt.close()



if __name__ == "__main__":
	for i in range(0, 7):
		print('i = {}'.format(i))
		main(i)
	#main(8)

