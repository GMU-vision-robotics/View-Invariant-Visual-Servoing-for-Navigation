"""Functions for working with RRTs."""
from collections import defaultdict, namedtuple
import cv2 # pylint: disable=wrong-import-order
import heapq
import math
import numpy as np # pylint: disable=wrong-import-order
import os
#import pdb
import pickle
import random

import bresenham
import meshcut

Bounds = namedtuple('Bounds', ['min_x', 'max_x', 'min_y', 'max_y'])

class PointConverter:
    """Convert coordinates in domain space to pixel space."""
    def __init__(self,
                 bounds,
                 px_per_meter,
                 padding_meters,
                 free,
                 ):
        (min_x,
         max_x,
         min_y,
         max_y) = bounds

        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.px_per_meter = px_per_meter
        self.padding_meters = padding_meters
        self.free = free

    def random_x(self):
        return random.random() * (self.max_x - self.min_x) + self.min_x

    def random_y(self):
        return random.random() * (self.max_y - self.min_y) + self.min_y

    def x_to_pixel(self, x):
        return int((x - self.min_x) * self.px_per_meter +
                   self.px_per_meter * self.padding_meters / 2.0)

    def y_to_pixel(self, y):
        return int((y - self.min_y) * self.px_per_meter +
                   self.px_per_meter * self.padding_meters / 2.0)

    def pixel_to_x(self, xpx):
        return ((float(xpx) / self.px_per_meter) - self.padding_meters / 2.0 + self.min_x)

    def pixel_to_y(self, ypx):
        return ((float(ypx) / self.px_per_meter) - self.padding_meters / 2.0 + self.min_y)

    def pixel_to_point(self,px):
        return self.pixel_to_x(px[0]), self.pixel_to_y(px[1])

    def point_to_pixel(self, p):
        return self.x_to_pixel(p[0]), self.y_to_pixel(p[1])

    def scaled_point_to_pixel(self, p, scale=1.0):
        return self.x_to_pixel(p[0])*scale, self.y_to_pixel(p[1])*scale

    def random_point(self, ):
        x = self.random_x()
        y = self.random_y()
        return x, y

    def free_point(self, x, y):
        x_px = self.x_to_pixel(x)
        y_px = self.y_to_pixel(y)
        # check image boundaries due to rounding errors
        return (x_px >= 0 and x_px < self.free.shape[1] and # pylint: disable=chained-comparison
                y_px >= 0 and y_px < self.free.shape[0] and
                self.free[y_px, x_px] == 255)

    def random_free_point(self, ):
        while True:
            x, y = self.random_point()
            if self.free_point(x, y):
                return x, y

    def get_bounds(self):
        return Bounds(self.min_x, self.max_x, self.min_y, self.max_y)

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

def get_cross_section(verts, faces, z=0.1):
    #z = np.min(verts[:, -1]) + z # 0.5 is the height of husky, actually it should be 0.37
    cross_section = meshcut.cross_section(verts, faces,
                                          plane_orig=(0, 0, z), plane_normal=(0, 0, 1))
    return cross_section

def cross_section_bounds(cross_section, padding_meters):
    return (min([x[:, 0].min() for x in cross_section]) - padding_meters/2.0,
            max([x[:, 0].max() for x in cross_section]) + padding_meters/2.0,
            min([x[:, 1].min() for x in cross_section]) - padding_meters/2.0,
            max([x[:, 1].max() for x in cross_section]) + padding_meters/2.0,)

def fill_in_gaps(lines):
    lines = [line for line in lines if len(line) > 1]
    endpoints = np.array([[l[0], l[-1]]  for l in lines]).reshape((len(lines)*2, 2))

    endpoint_distances = np.sqrt(np.sum(np.power(
        (endpoints.reshape(endpoints.shape[0], 1, 2) -
         endpoints.reshape(1, endpoints.shape[0], 2)), 2), axis=2))
    sorted_endpoint_indexes = np.argsort(endpoint_distances)

    connections = {}
    fix_up_lines = []

    for i in range(len(lines)):
        for j in [i * 2, i * 2 + 1]:
            if j not in connections:
                for p in sorted_endpoint_indexes[j]:
                    if p != j:
                        connections[j] = p
                        connections[p] = j
                        l0 = j//2
                        idx0 = 0
                        if j%2 == 1:
                            idx0 = -1
                        l1 = p//2
                        idx1 = 0
                        if p%2 == 1:
                            idx1 = -1
                        fix_up_lines.append(np.array([lines[l0][idx0],
                                                      lines[l1][idx1]]))
                        break
    return fix_up_lines

def make_free_space_image(cross_section_2d,
                          px_per_meter,
                          padding_meters,
                          erosion_iterations=5):

    min_x, max_x, min_y, max_y = cross_section_bounds(cross_section_2d, padding_meters)

    image_height = int(np.ceil(px_per_meter * (max_x - min_x) +
                               px_per_meter * padding_meters))
    image_width = int(np.ceil(px_per_meter * (max_y - min_y) +
                              px_per_meter * padding_meters))

    x_adj = -min_x * px_per_meter + px_per_meter * padding_meters / 2.0
    y_adj = -min_y * px_per_meter + px_per_meter * padding_meters / 2.0

    lines = [np.round(x * px_per_meter + np.array([x_adj, y_adj]), 0).astype(np.int32)
             for x in cross_section_2d]

    image = np.zeros((image_width, image_height, 3), np.uint8)
    cv2.polylines(image, lines, False, (255, 255, 255), thickness=3)

    cv2.polylines(image, fill_in_gaps(lines), False, (255, 255, 255), thickness=3)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = np.zeros((image_width + 2, image_height + 2), np.uint8)
    cv2.floodFill(gray, mask, (2000, 2000), 255)

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(gray, kernel, iterations=erosion_iterations)
    return image, erosion

## return True when there is no obstacle
def line_check(p0, p1, free, skip=0):
    p_count = 0
    for p in bresenham.line_points(p0, p1):
        p_count += 1
        if p_count <= skip:
            continue
        #print('p[0] = {}, p[1] = {}'.format(p[0], p[1]))
        #print('free[{}, {}] = {}'.format(p[1], p[0], free[p[1], p[0]]))
        if free[p[1], p[0]] != 255:
            return False
    return True

def make_rrt(
        cross_section,
        padding_meters,
        px_per_meter,
        num_nodes,
        epsilon,
        free,
        extend_line=False,
        crossing_paths=True,
        new_edge_callback=None,
        ):
    if not crossing_paths:
        free = np.copy(free)

    min_x, max_x, min_y, max_y = cross_section_bounds(cross_section, padding_meters)

    pc = PointConverter(
        (min_x, max_x, min_y, max_y),
        px_per_meter, padding_meters, free)

    starting_point = pc.random_free_point()
    nodes_x = np.zeros(num_nodes, np.float32)
    nodes_y = np.zeros(num_nodes, np.float32)
    nodes_x[0] = starting_point[0]
    nodes_y[0] = starting_point[1]
    edges = np.full((num_nodes-1, 2), -1)
    edges_from_px = []
    edges_to_px = []

    i = 1
    misses = 0
    while True:
        next_node = pc.random_point()
        distances = np.sqrt(np.power(nodes_x[:i] -
                                     next_node[0], 2) +
                            np.power(nodes_y[:i] -
                                     next_node[1], 2))

        closest_point_idx = np.argmin(distances)
        theta = math.atan2(next_node[1] -
                           nodes_y[closest_point_idx], next_node[0] -
                           nodes_x[closest_point_idx])

        x_delta = np.cos(theta) * epsilon
        y_delta = np.sin(theta) * epsilon

        node_iter = 1
        p0 = pc.point_to_pixel((nodes_x[closest_point_idx],
                                nodes_y[closest_point_idx]))
        while True:
            node = (nodes_x[closest_point_idx] + x_delta * node_iter,
                    nodes_y[closest_point_idx] + y_delta * node_iter)
            node_iter += 1
            p1 = pc.point_to_pixel(node)

            if pc.free_point(*node) and line_check(p0, p1, free, skip=5):
                nodes_x[i] = node[0]
                nodes_y[i] = node[1]
                edges[i-1][0] = closest_point_idx
                edges[i-1][1] = i
                if new_edge_callback:
                    new_edge_callback(nodes_x, nodes_y, edges)
                edges_from_px.append(p0)
                edges_to_px.append(p1)
                if not crossing_paths:
                    cv2.line(free, edges_from_px[i-1],
                             edges_to_px[i-1], (0), thickness=3)

                p0 = p1
                i += 1
                if i%1000 == 0:
                    pass
                if not extend_line or i == num_nodes:
                    break
            else:
                misses += 1
                if misses%1000 == 0:
                    pass
                break
        if i == num_nodes:
            break

    # Connect leaf nodes to closest points
    leaf_index = np.setdiff1d(edges[:, 1], edges[:, 0])
    leaf_nodes = np.stack((nodes_x[leaf_index], nodes_y[leaf_index]), axis=1)
    distances = np.sqrt(np.sum(np.power((leaf_nodes.reshape(leaf_nodes.shape[0],
                                                            1,
                                                            2) -
                                         leaf_nodes.reshape(1,
                                                            leaf_nodes.shape[0],
                                                            2)),
                                        2),
                               axis=2))
    sorted_endpoint_indexes = np.argsort(distances)

    connect_leaves = False
    if connect_leaves:
        new_edges = defaultdict(set)
        for i in range(len(leaf_nodes)):
            node0_idx = leaf_index[i]
            p0 = pc.point_to_pixel((nodes_x[node0_idx], nodes_y[node0_idx]))
            for j in list(sorted_endpoint_indexes[i, 1:]):
                node1_idx = leaf_index[j]
                if node1_idx not in new_edges[node0_idx]:
                    p1 = pc.point_to_pixel((nodes_x[node1_idx], nodes_y[node1_idx]))
                    if line_check(p0, p1, free):
                        new_edges[node0_idx].add(node1_idx)
                        new_edges[node1_idx].add(node0_idx)
                        edges_from_px.append(pc.point_to_pixel((nodes_x[node0_idx],
                                                                nodes_y[node0_idx])))
                        edges_to_px.append(pc.point_to_pixel((nodes_x[node1_idx],
                                                              nodes_y[node1_idx])))
                        break
        new_edge_array = np.array([[i, j]
                                   for i in new_edges
                                   for j in new_edges[i]])
        edges = np.vstack((edges, new_edge_array))

    return edges_from_px, edges_to_px, nodes_x, nodes_y, edges

class HeapNode:
    """Node for heap used in A* algorithm AStarHeap."""
    def __init__(self, estimated, cost, state, path):
        self.estimated = estimated
        self.cost = cost
        self.state = state
        self.path = path
        self.deleted = False
        self.start_node = None
        self.goal_node = None

    def __lt__(self, other):
        return self.estimated < other.estimated

class AStarHeap:
    """Heap used for A*, with deletion and replacement."""
    def __init__(self):
        self.heap = []
        self.state_nodes = {}
        self.size = 0

    def heappush(self, node):
        heapq.heappush(self.heap, node)
        self.state_nodes[node.state] = node
        self.size += 1

    def exists_worse(self, state, estimated):
        if state in self.state_nodes and self.state_nodes[state].estimated > estimated:
            return True
        return False

    def exists(self, state):
        return state in self.state_nodes

    def replace(self, new_node):
        self.state_nodes[new_node.state].deleted = True
        del self.state_nodes[new_node.state]
        self.size -= 1
        self.heappush(new_node)

    def heappop(self):
        while self.heap[0].deleted:
            heapq.heappop(self.heap)
        self.size -= 1
        node = heapq.heappop(self.heap)
        del self.state_nodes[node.state]
        return node

    def getsize(self):
        return self.size

def A_star_search(start_node, goal_node, goal_distances, edges):
    frontier = AStarHeap()
    frontier.heappush(HeapNode(goal_distances[start_node], 0, start_node, [start_node]))
    explored = set()

    while True:
        if not frontier.getsize():
            return None
        node = frontier.heappop()
        if node.state == goal_node:
            node.start_node = start_node
            node.goal_node = goal_node
            return node
        explored.add(node.state)
        for child in edges[node.state].keys():
            newnode = HeapNode(node.cost + goal_distances[child],
                               node.cost + edges[node.state][child],
                               child,
                               node.path + [child])
            if (not (newnode.state in explored or
                     frontier.exists(newnode.state))):
                frontier.heappush(newnode)
            elif frontier.exists_worse(newnode.state,
                                       newnode.estimated):
                frontier.replace(newnode)
    return None

class PathFinder:
    """Finds path in RRT using A* search."""
    def __init__(
            self,
            directory=None,
            nodes_x=None,
            nodes_y=None,
            edges_idx=None,
            free=None,
            config=None,
            pc=None,
            cross_section_2d=None,
            edges=None,
            ):
        self.dir = directory
        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        self.edges_idx = edges_idx
        self.free = free
        self.config = config
        self.pc = pc
        self.cross_section_2d = cross_section_2d
        self.edges = edges

    def load(self):
        rrt_file_name = os.path.join(self.dir, 'rrt.npz')
        config_file_name = os.path.join(self.dir, 'config.pickle')

        npfile = np.load(rrt_file_name)
        self.nodes_x = npfile['arr_0']
        self.nodes_y = npfile['arr_1']
        self.edges_idx = npfile['arr_2']
        self.free = npfile['arr_3']
        self.cross_section_2d = [npfile[x] for x in npfile.keys()
                                 if int(x.replace('arr_', '')) > 4]

        with open(config_file_name, 'rb') as config_file:
            self.config = pickle.load(config_file)
        self.pc = PointConverter((
            self.config['min_x'],
            self.config['max_x'],
            self.config['min_y'],
            self.config['max_y']),
            self.config['px_per_meter'],
            self.config['padding_meters'],
            self.free)

        edges = defaultdict(dict)
        edges_idx = self.edges_idx
        nodes_x = self.nodes_x
        nodes_y = self.nodes_y
        for edge_idx in edges_idx:
            n0 = edge_idx[0]
            n1 = edge_idx[1]
            distance = math.sqrt((nodes_x[n0] - nodes_x[n1])**2 +
                                 (nodes_y[n0] - nodes_y[n1])**2)
            edges[n0][n1] = distance
            edges[n1][n0] = distance
        self.edges = edges

    def find(self, x0, y0, x1, y1):
        try:
            pc = self.pc
            nodes_x = self.nodes_x
            nodes_y = self.nodes_y

            if not pc.free_point(x0, y0):
                raise Exception("starting point ({}, {}) is not free".format(x0, y0))

            if not pc.free_point(x1, y1):
                raise Exception("starting point ({}, {}) is not free".format(x1, y1))

            start_node = self.node_closest_to_point(x0, y0)
            goal_node = self.node_closest_to_point(x1, y1)
            goal_distances = np.sqrt(np.power(nodes_x - nodes_x[goal_node], 2) +
                                     np.power(nodes_y - nodes_y[goal_node], 2))

            solution = A_star_search(start_node, goal_node, goal_distances, self.edges)

            lines = [np.array([pc.y_to_pixel(y0), pc.x_to_pixel(x0)])]
            lines += [np.array([pc.y_to_pixel([nodes_y[node]]), pc.x_to_pixel(nodes_x[node])])
                      for node in solution.path]
            lines += [np.array([pc.y_to_pixel(y1), pc.x_to_pixel(x1)])]

            return solution, lines
        except Exception as e: # pylint: disable=broad-except
            print(e)
            return None, None

    def node_closest_to_point(self, x, y):
        distances = np.argsort(np.sqrt(np.power(self.nodes_x - x, 2) +
                                       np.power(self.nodes_y - y, 2)))
        node = None
        for i in range(distances.shape[0]):
            j = distances[i]
            if line_check(self.pc.point_to_pixel((self.nodes_x[j],
                                                  self.nodes_y[j])),
                          self.pc.point_to_pixel((x, y)), self.free):
                node = j
                break
        if node is None:
            raise Exception("Could not find a clear path from ({}, {}) to a node".format(x, y))
        return node

    def get_bounds(self):
        return self.pc.get_bounds()

    ## added by myself
    def pixel_to_point(self, p):
        return self.pc.pixel_to_point(p)
    ## added by myself
    def point_to_pixel(self, p):
        return self.pc.point_to_pixel(p)
