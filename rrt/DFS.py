import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

class graph:
	def __init__(self, np_file):
		self.v_lst = []
		for vertex in np_file['vertices']:
			x, y = vertex[0], vertex[1]
			self.v_lst.append((x*10, y*10))

		self.G = nx.Graph()
		for i in range(len(self.v_lst)):
			self.G.add_node(i)
		for edge in np_file['edges']:
			e1, e2 = edge
			self.G.add_edge(e1, e2)

		'''
		plt.subplot(111)
		nx.draw(self.G, with_labels=True, font_weight='bold')
		plt.show()
		'''

	def idx_to_node(self, idx):
		return self.v_lst[idx]


	def find_paths(self, start_node, end_node):
		# localize start and end_node index
		dist_start_node = np.zeros(len(self.v_lst))
		x1, y1 = start_node
		for i, v in enumerate(self.v_lst):
			#print('v = {}'.format(v))
			x2, y2 = v
			dist = (x1 - x2)**2 + (y1 - y2)**2
			dist_start_node[i] = dist
		start_node_idx = np.argmin(dist_start_node)

		dist_end_node = np.zeros(len(self.v_lst))
		x1, y1 = end_node
		for i, v in enumerate(self.v_lst):
			x2, y2 = v
			dist = (x1 - x2)**2 + (y1 - y2)**2
			dist_end_node[i] = dist
		end_node_idx = np.argmin(dist_end_node)

		print('matched start_node = {}, matched end_node = {}'.format(self.v_lst[start_node_idx], self.v_lst[end_node_idx]))

		paths = [p for p in nx.all_shortest_paths(self.G, start_node_idx, end_node_idx)]
		node_paths = []
		for p in paths:
			node_path = []
			for idx in p:
				node_path.append(self.idx_to_node(idx))
			node_paths.append(node_path)

		return node_paths



def diffDriveT(xi, v, omega, deltaT, tmax):
    arrayX = []
    arrayY = []
    theta = []
    arrayX.append(xi[0])
    arrayY.append(xi[1])
    theta.append(xi[2])
    for i in range(tmax):
        arrayX.append(arrayX[i] + v*deltaT*math.cos(theta[i]))
        arrayY.append(arrayY[i] + v*deltaT*math.sin(theta[i]))
        theta.append(theta[i] + omega*deltaT)
    # print(arrayX)
    return arrayX, arrayY, theta