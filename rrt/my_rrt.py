from collections import defaultdict
import heapq
import math
import numpy as np
import os
import random
from math import floor

class PointConverter:
    """Convert coordinates in domain space to pixel space."""
    def __init__(self,
                 free,
                 px_per_meter=500
                 ):

        self.free = free
        h, w = free.shape
        self.px_per_meter = px_per_meter
        self.min_x = 0
        self.max_x = int(w/self.px_per_meter)
        self.min_y = 0
        self.max_y = int(h/self.px_per_meter)

    def random_x(self):
        return random.random() * (self.max_x - self.min_x) + self.min_x

    def random_y(self):
        return random.random() * (self.max_y - self.min_y) + self.min_y

    def x_to_pixel(self, x):
        return int((x - self.min_x) * self.px_per_meter)

    def y_to_pixel(self, y):
        return int((y - self.min_y) * self.px_per_meter)

    def pixel_to_x(self, xpx):
        return float(xpx) / self.px_per_meter

    def pixel_to_y(self, ypx):
        return float(ypx) / self.px_per_meter

    def pixel_to_point(self,px):
        return self.pixel_to_x(px[0]), self.pixel_to_y(px[1])

    def point_to_pixel(self, p):
        return self.x_to_pixel(p[0]), self.y_to_pixel(p[1])

    def random_point(self, ):
        x = self.random_x()
        y = self.random_y()
        return x, y

    def free_point(self, x, y):
        x_px = self.x_to_pixel(x)
        #print('x_px = {}'.format(x_px))
        y_px = self.y_to_pixel(y)
        #print('y_px = {}'.format(y_px))
        return (x_px >= 0 and x_px < self.free.shape[1] and
                y_px >= 0 and y_px < self.free.shape[0] and
                self.free[y_px, x_px] == 255)

    def random_free_point(self, ):
        while True:
            x, y = self.random_point()
            if self.free_point(x, y):
                return x, y

## return True when there is no obstacle
def line_check(p0, p1, free, skip=0):
    p_count = 0
    for p in line_points(p0, p1):
        p_count += 1
        if p_count <= skip:
            continue
        #print('p[0] = {}, p[1] = {}'.format(p[0], p[1]))
        #print('free[{}, {}] = {}'.format(p[1], p[0], free[p[1], p[0]]))
        if free[p[1], p[0]] != 255:
            return False
    return True

def make_rrt(free, num_nodes=5000, epsilon=.1):
    pc = PointConverter(free)

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
        #print('next_node = {}'.format(next_node))
        distances = np.sqrt(np.power(nodes_x[:i] - next_node[0], 2) + np.power(nodes_y[:i] - next_node[1], 2))

        closest_point_idx = np.argmin(distances)
        theta = math.atan2(next_node[1] - nodes_y[closest_point_idx], 
                            next_node[0] - nodes_x[closest_point_idx])

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
                
                edges_from_px.append(p0)
                edges_to_px.append(p1)

                p0 = p1
                i += 1
                if i%1000 == 0:
                    pass
                if True:
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
    distances = np.sqrt(np.sum(np.power((leaf_nodes.reshape(leaf_nodes.shape[0], 1, 2) -
                                         leaf_nodes.reshape(1, leaf_nodes.shape[0], 2)),
                                        2), axis=2))
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
                        edges_from_px.append(pc.point_to_pixel((nodes_x[node0_idx], nodes_y[node0_idx])))
                        edges_to_px.append(pc.point_to_pixel((nodes_x[node1_idx], nodes_y[node1_idx])))
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
                               child, node.path + [child])
            if (not (newnode.state in explored or frontier.exists(newnode.state))):
                frontier.heappush(newnode)
            elif frontier.exists_worse(newnode.state, newnode.estimated):
                frontier.replace(newnode)
    return None

class PathFinder:
    """Finds path in RRT using A* search."""
    def __init__(
            self,
            nodes_x=None,
            nodes_y=None,
            edges_idx=None,
            free=None,
            ):
        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        self.edges_idx = edges_idx
        self.free = free
        self.pc = PointConverter(self.free)
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

    def find_path_between_points(self, x0, y0, x1, y1):
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
            lines += [np.array([pc.y_to_pixel(nodes_y[node]), pc.x_to_pixel(nodes_x[node])])
                      for node in solution.path]
            lines += [np.array([pc.y_to_pixel(y1), pc.x_to_pixel(x1)])]

            return solution, lines
        except Exception as e: # pylint: disable=broad-except
            print(e)
            return None, None

    def find_path_between_pixels(self, p1, p2):
        
        pc = self.pc
        nodes_x = self.nodes_x
        nodes_y = self.nodes_y

        x0, y0 = self.pixel_to_point(p1)
        x1, y1 = self.pixel_to_point(p2)

        if not pc.free_point(x0, y0):
            raise Exception("starting point ({}, {}) is not free".format(x0, y0))

        if not pc.free_point(x1, y1):
            raise Exception("starting point ({}, {}) is not free".format(x1, y1))

        start_node = self.node_closest_to_point(x0, y0)
        goal_node = self.node_closest_to_point(x1, y1)
        goal_distances = np.sqrt(np.power(nodes_x - nodes_x[goal_node], 2) +
                                 np.power(nodes_y - nodes_y[goal_node], 2))

        solution = A_star_search(start_node, goal_node, goal_distances, self.edges)
        #print('solution.path = {}'.format(solution.path))

        lines = [np.array([pc.y_to_pixel(y0), pc.x_to_pixel(x0)])]
        lines += [np.array([pc.y_to_pixel(nodes_y[node]), pc.x_to_pixel(nodes_x[node])])
                  for node in solution.path]
        lines += [np.array([pc.y_to_pixel(y1), pc.x_to_pixel(x1)])]

        return solution, lines


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

    def pixel_to_point(self, p):
        return self.pc.pixel_to_point(p)
    
    def point_to_pixel(self, p):
        return self.pc.point_to_pixel(p)



def slope(p1, p2):
    if p1[0] == p2[0]:
        raise ValueError("Undefined slope for points ({}, {}) and ({}, {})".format(p1[0], p1[1], p2[0], p2[1]))
    return (float(p2[1])-float(p1[1]))/(float(p2[0])-float(p1[0]))

def line_octant(p1, p2):
    m = slope(p1, p2)

    if 0 <= m <= 1 and p1[0] < p2[0]:
        return 0

    if m > 1 and p1[1] < p2[1]:
        return 1

    if m < -1 and p1[1] < p2[1]:
        return 2

    if 0 >= m >= -1 and p2[0] < p1[0]:
        return 3

    if 0 < m <= 1 and p2[0] < p1[0]:
        return 4

    if m > 1 and p2[1] < p1[1]:
        return 5

    if -1 > m and p2[1] < p1[1]: 
        return 6

    if 0 > m >= -1 and p1[0] < p2[0]:
        return 7

    raise Exception("Unknown quadrant for points ({}, {}) and ({}, {})".format(p1[0], p1[1], p2[0], p2[1]))

octant_func_from = [
    lambda p: (p[0]  ,p[1]),
    lambda p: (p[1]  ,p[0]),
    lambda p: (p[1]  ,-p[0]),
    lambda p: (-p[0] ,p[1]),
    lambda p: (-p[0] ,-p[1]),
    lambda p: (-p[1] ,-p[0]),
    lambda p: (-p[1] ,p[0]),
    lambda p: (p[0]  ,-p[1]),
    ]

def get_octant_func_from(p1, p2):
    return octant_func_from[line_octant(p1, p2)]

octant_func_to = [
    lambda p: (p[0]  ,p[1]),
    lambda p: (p[1]  ,p[0]),
    lambda p: (-p[1] ,p[0]),
    lambda p: (-p[0] ,p[1]),
    lambda p: (-p[0] ,-p[1]),
    lambda p: (-p[1] ,-p[0]),
    lambda p: (p[1]  ,-p[0]),
    lambda p: (p[0]  ,-p[1]),
    ]

def get_octant_func_to(p1, p2):
    return octant_func_to[line_octant(p1, p2)]

def line_points(p1, p2):
    if p1[0] != p2[0]:
        from_func = get_octant_func_from(p1, p2)
        to_func   = get_octant_func_to(p1, p2)
        p1 = from_func(p1)
        p2 = from_func(p2)
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        x  = p1[0]
        y  = p1[1]
        eps = 0

        while True:
            yield to_func((x,y))
            x += 1
            if x > p2[0]:
                raise StopIteration
            eps += dy
            if eps * 2 >= dx:
                y += 1
                eps -= dx
    else:
        step = adj  = 1
        if p1[1] > p2[1]:
            step = adj = -1

        for y in range(p1[1], p2[1] + adj, step):
            yield (p1[0],y)
