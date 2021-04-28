import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import math
from scipy import ndimage
import cv2
import my_code.rrt.rrt as rrt

# 500 pixels is one meter
show_animation = False

class Node:
    """
    Node class for dijkstra search
    """
    
    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

        
    def __str__(self):

        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.parent_index)

def generateKNeighbors(x, y, k, arrayX, arrayY, img, img_dist):
    # x, y = starting point
    # k = number of neighbors
    neighbors = []
    distances = []
    dist = 0
    length = len(arrayX)
    if (img[y + 1][x] != 0) and (img_dist[y+1][x] > k):
        neighbors.append((x, y + 1))
    if (img[y - 1][x] != 0) and (img_dist[y-1][x] > k):
        neighbors.append((x, y - 1))
    if (img[y][x - 1] != 0) and (img_dist[y][x-1] > k):
        neighbors.append((x - 1, y))
    if (img[y][x + 1] != 0) and (img_dist[y][x+1] > k):
        neighbors.append((x + 1, y))
    if (img[y - 1][x - 1] != 0) and (img_dist[y-1][x-1] > k):
        neighbors.append((x - 1, y - 1))
    if (img[y + 1][x + 1] != 0) and (img_dist[y+1][x+1] > k):
        neighbors.append((x + 1, y + 1))
    return neighbors

def distance(x1, y1, x2, y2):
    return math.sqrt(((x1-x2)**2)+((y1-y2)**2))
    # return abs(x1-x2) + abs(y1-y2)

def generate_road_map(x, y, rr, img, img_dist):
    #x, y = arrays of sample points
    #rr = robot radius (5)
    #Generating the roadmap proceeds by going through all samples
    #for each sample finding k - nearest neighbours for each edge between a sample and it’s neighbours
    #check whether that edges is in free space if is is append the edge to the graph
    #1000 indices: 1000 lists with each indice attached to a list
    roadmap = dict()
    neighbors = []
    newNum = len(x)
    check = False
    counter = 0
    # print(newNum)
    for i in range(newNum):
      edge_id = []
      neighbors = generateKNeighbors(x[i], y[i], rr, x, y, img, img_dist)
      counter += 1
      for j in neighbors:
        edge_id.append((j[0], j[1]))
      roadmap[(x[i], y[i])] = edge_id
    return roadmap

def dijkstra(sx, sy, gx, gy, road_map):
    rx = []
    ry = []
    open_set, closed_set = dict(), dict() #open_set is the unvisited set, closed_set is the visited set
    path_found = True
    start_node = Node(sx, sy, 0.0, (-1, -1))
    goal_node = Node(gx, gy, 0.0, (-1, -1))
    open_set[(sx, sy)] = start_node
    path_found = True
    while True:
        # print(open_set)
        # print("Goal: ", gx, gy)
        if not open_set:
            print("Cannot find path")
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        if c_id == (gx, gy):
            print("goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break

        for i in road_map[(current.x, current.y)]:
            dx = i[0] - current.x
            dy = i[1] - current.y
            # d = math.hypot(dx, dy)
            d = abs(dx) + abs(dy)
            # print(current.x, current.y)
            tempnode = Node(i[0], i[1], current.cost + d, (current.x, current.y))
            if (i[0], i[1]) in closed_set:
                continue
            if (i[0], i[1]) not in open_set.keys():
                open_set[(i[0], i[1])] = tempnode
            elif tempnode.cost < open_set[(i[0], i[1])].cost:
                open_set[(i[0], i[1])].cost = tempnode.cost
                open_set[(i[0], i[1])].parent_index = c_id
        del open_set[c_id]
        closed_set[c_id] = current

    rx, ry = [], []
    if path_found == False:
        return [], []

    parent_index = goal_node.parent_index
    while parent_index != (-1, -1):
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index
    return rx, ry

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

directory = '/home/yimeng/Datasets/GibsonEnv_old/gibson/assets/dataset/Uvalda_for_rrt'
path_finder = rrt.PathFinder(directory)
print("Loading...")
path_finder.load()
print("Finding...")


img = path_finder.free.copy()
h_ori, w_ori = img.shape

img = cv2.resize(img, (int(w_ori*0.05), int(h_ori*0.05)), interpolation=cv2.INTER_NEAREST)
h_cur, w_cur = img.shape
ratio_w, ratio_h = w_ori/w_cur, h_ori/h_cur
#assert 1==2


img_dist = ndimage.distance_transform_edt(img)


height, width = img.shape[0], img.shape[1]
print(height, width)
height, width = img_dist.shape[0], img_dist.shape[1]
# print(height, width)


# test start and goal positions in pixel coordinates in the image
'''
sx = 100
sy = 50
gx = 235
gy = 225
'''

sx = 150
sy = 155
gx = 250
gy = 155

'''
sx = 225
sy = 175
gx = 297
gy = 175
'''




# print(img_dist[sy][sx])
# print(img_dist[gy][gx])

# generate different path varying the clearance and plot them in distance transf image
dict_rx_ry = {}
for robot_size in range(5):  #5
    filteredX = []
    filteredY = []
    for i in range(height):
        for j in range(width):
            # if (img[i][j] != 0):
            if (img[i][j] != 0) and (img_dist[i][j] > robot_size):
                filteredX.append(j)
                filteredY.append(i)
                # plt.plot(j,i,'.')
    
    roadmap = generate_road_map(filteredX, filteredY, robot_size, img, img_dist)
    print('roadmap done')
    rx, ry = dijkstra(sx, sy, gx, gy, roadmap)
    #for i in range(1, len(rx)):
    #  plt.plot([rx[i-1], rx[i]], [ry[i-1], ry[i]], 'b-')
      # print(rx[i],ry[i])
      # plt.show()
    dict_rx_ry[robot_size] = {}
    dict_rx_ry[robot_size]['rx'] = rx
    dict_rx_ry[robot_size]['ry'] = ry
#assert 1==2

dict_poses = {}
for robot_size in range(5):
    print('================================= robot_size = {} ============================'.format(robot_size))
    dict_poses[robot_size] = []

    rx = dict_rx_ry[robot_size]['rx']
    ry = dict_rx_ry[robot_size]['ry']

    # visual the last set of waypoints in high-res image      
    rx_up = [x * ratio_w for x in rx]
    ry_up = [x * ratio_h for x in ry]

    subsample = 10  #pick every 10th point

    # set linear velocity and follow the set of waypoints
    v = 500
    x0 = rx_up[0]
    y0 = -ry_up[0]
    for i in range(1, (int)(len(rx)/subsample)):
        xg = rx_up[subsample * i]
        yg = -ry_up[subsample * i]

        th0 = math.atan2((yg - y0), (xg -x0))
        s0 = [x0, y0, th0]

        err_dist = 50
        err = math.sqrt((xg - x0) ** 2 + (yg - y0) ** 2)
        while err > err_dist:
            w = math.atan2(yg -y0, xg -x0) - th0
            xn, yn, thn = diffDriveT(s0, v, w , 0.1, 1)
            err = math.sqrt((xg - x0)**2 + (yg - y0)**2)
            print('err', err, xn[1], yn[1], thn[1])

            x, y = path_finder.pixel_to_point((xn[1], -yn[1]))

            dict_poses[robot_size].append((x, y, thn[1]))

            s0 = [xn[1], yn[1], thn[1]]
            x0 = xn[1]
            y0 = yn[1]
            th0 = thn[1]




np.save('my_code/rrt/JK_poses.npy', dict_poses)

