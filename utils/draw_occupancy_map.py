import sys
import matplotlib.pyplot as plt
import meshcut
import numpy as np

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

def main():
	fn = args.meshFilePath
	verts, faces = load_obj(fn)
	z =  np.min(verts[:,-1]) + 0.5 # 0.5 is the height of husky, actually it should be 0.37
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
	plt.xticks(np.arange(-2, 10, 0.5))
	plt.yticks(np.arange(-2, 9, 0.5))
	plt.grid(True)
	## save image to the path
	if args.save_name is not None:
		print('save image to {}'.format(args.save_name))
		plt.savefig(args.save_name)
	else:
		plt.show()


if __name__ == "__main__":	
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--meshFilePath', type=str, default='gibson/assets/dataset/Allensville/mesh_z_up.obj')
	parser.add_argument('--save_name', type=str, default=None)
	args = parser.parse_args()
	main()

