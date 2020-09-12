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
	'''
	# find better z, because the scene might have numbers of floors
	if args.trajFilePath:
		print('Process trajectory file: {}'.format(args.trajFilePath))
		infos = np.load(args.trajFilePath)
		for i in range(len(infos)):
			first = infos[i]['eye_pos']
			if z > first[2]:
				z = first[2]
	'''
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
	print('draw trajectory ...')
	if args.trajFilePath:
		print('Process trajectory file: {}'.format(args.trajFilePath))
		infos = np.load(args.trajFilePath)
		for i in range(len(infos)-1):
			# infos is of structure [{'eye_pos':, 'eye_quat':, 'episode':}, {}]
			first = infos[i]['eye_pos']
			second = infos[i+1]['eye_pos']
			plt.plot([first[0], second[0]], [first[1], second[1]], 'k')
		# draw start point as yellow circle and end point as yellow star
		start_point = infos[0]['eye_pos']
		end_point = infos[len(infos) - 1]['eye_pos']
		plt.plot(start_point[0], start_point[1], color='r', marker='o')
		plt.plot(end_point[0], end_point[1], color='r', marker='*')
	#plt.show()
	#print('start_point: {}'.format(start_point))
	#print('end_point: {}'.format(end_point))
	## draw target on image
	if args.targetX is not None and args.targetY is not None:
		print('draw target to image ...')
		plt.plot(args.targetX, args.targetY, color='y', marker='*')
	## save image to the path
	plt.title('{}'.format(args.save_name[args.save_name.rfind('/')+1 : -4]))
	if args.save_name is not None:
		print('save image to {}'.format(args.save_name))
		plt.savefig(args.save_name)
	else:
		plt.show()


if __name__ == "__main__":	
	import argparse
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--meshFilePath', type=str, default='gibson/assets/dataset/Allensville/mesh_z_up.obj')
	parser.add_argument('--trajFilePath', type=str, default=None)#'my_code/hand_manipulate/obs_Allensville_infos.npy')
	parser.add_argument('--targetX', type=float, default=None)
	parser.add_argument('--targetY', type=float, default=None)
	parser.add_argument('--save_name', type=str, default=None)
	args = parser.parse_args()
	main()

