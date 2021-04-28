import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import math
from scipy import ndimage
import cv2

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

'''
class Node:
    """
    Node class for dijkstra search
    """
    
    def __init__(self, x, y, cost, parent_x, parent_y):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_x = parent_x
        self.parent_y = parent_y
        
    def __str__(self):
        return('point = ({}, {}), cost = {}, parent = ({}, {})'.format(self.x, self.y, self.cost, self.parent_x, self.parent_y))
'''

def generateKNeighbors(x, y, k, arrayX, arrayY, img, img_dist):
    # x, y = starting point
    # k = number of neighbors
    neighbors = []
    distances = []
    dist = 0
    length = len(arrayX)
    if (img[y + 1][x] != 0) and (img_dist[y + 1][x] > k):
        neighbors.append((x, y + 1))
    if (img[y - 1][x] != 0) and (img_dist[y - 1][x] > k):
        neighbors.append((x, y - 1))
    if (img[y][x - 1] != 0) and (img_dist[y][x - 1] > k):
        neighbors.append((x - 1, y))
    if (img[y][x + 1] != 0) and (img_dist[y][x + 1] > k):
        neighbors.append((x + 1, y))
    if (img[y - 1][x - 1] != 0) and (img_dist[y - 1][x - 1] > k):
        neighbors.append((x - 1, y - 1))
    if (img[y + 1][x + 1] != 0) and (img_dist[y + 1][x + 1] > k):
        neighbors.append((x + 1, y + 1))
    if (img[y - 1][x + 1] != 0) and (img_dist[y - 1][x + 1] > k):
        neighbors.append((x + 1, y - 1))
    if (img[y + 1][x - 1] != 0) and (img_dist[y + 1][x - 1] > k):
        neighbors.append((x - 1, y + 1))
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
    newNum = len(x)
    for i in range(newNum):
      edge_id = []
      neighbors = generateKNeighbors(x[i], y[i], rr, x, y, img, img_dist)
      roadmap[(x[i], y[i])] = neighbors
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

scene = 'Allensville'

img = cv2.imread('gibson/assets/dataset/{}_for_rrt/free.png'.format(scene), 0)
#img_dist = mpimg.imread('uvalda_dist.png')
# imgd = Image.open('uvalda_05_dist.png')
# img_dist = np.array(imgd)
h, w = img.shape
img = cv2.resize(img, (int(w*0.05), int(h*0.05)), interpolation=cv2.INTER_NEAREST)
#assert 1==2


img_dist = ndimage.distance_transform_edt(img)
# imgtemp = ndimage.distance_transform_edt(img_dist)
# replace img_dist with imgtemp
# use function to turn the img into binary
# then invert the image
plot1 = plt.figure(1)
plt.imshow(img)


height, width = img.shape

plot2 = plt.figure(2);
plt.imshow(img_dist)

# test start and goal positions in pixel coordinates in the image
sx = 150
sy = 90
gx = 160
gy = 180

''' 
# Uvalda
sx = 150
sy = 155
gx = 250
gy = 155
'''



plt.plot([sx], [sy], 'r*')
plt.plot([gx], [gy], 'g*')
#plt.show()
# print(img_dist[sy][sx])
# print(img_dist[gy][gx])

# generate different path varying the clearance and plot them in distance transf image
for robot_size in range(0, 20, 3):  #5
    try:
        mask_free_space = np.logical_and((img != 0), (img_dist > robot_size))
        x_coords = np.linspace(0, width-1, width)
        y_coords = np.linspace(0, height-1, height)
        all_coords = np.meshgrid(x_coords, y_coords)
        x_coords = all_coords[0].astype('int')
        y_coords = all_coords[1].astype('int')
        x_coords = x_coords[mask_free_space]
        y_coords = y_coords[mask_free_space]
      
        roadmap = generate_road_map(x_coords, y_coords, robot_size, img, img_dist)
        print('roadmap done')
        rx, ry = dijkstra(sx, sy, gx, gy, roadmap)
        for i in range(1, len(rx)):
          plt.plot([rx[i-1], rx[i]], [ry[i-1], ry[i]], 'b-')
    except:
        print('Can not do this')
      # print(rx[i],ry[i])
      # plt.show()
plt.show()
assert 1==2

# visual the last set of waypoints in high-res image      
imgfull = cv2.imread('gibson/assets/dataset/Uvalda_for_rrt/free.png', 0);
rx_up = [x * 20 for x in rx]
ry_up = [x * 20 for x in ry]

sz = len(rx_up)-1
plot3 = plt.figure(3)
plt.imshow(imgfull)
subsample = 10  #pick every 10th point
plt.plot(rx_up[0], ry_up[0], 'r*')
plt.plot(rx_up[sz], ry_up[sz], 'g*')
print('*********', rx_up[sz], ry_up[sz])
for i in range(1, (int)(len(rx)/subsample)):
     plt.plot(rx_up[subsample*i], ry_up[subsample*i], 'b.')

# set linear velocity and follow the set of waypoints
v = 500
x0 = rx_up[0]
y0 = -ry_up[0]
for i in range(1,(int)(len(rx)/subsample)):
    xg = rx_up[subsample * i]
    yg = -ry_up[subsample * i]

    th0 = math.atan2((yg - y0), (xg -x0))
    s0 = [x0, y0, th0]
    # plt.plot(x0[0], x0[1], "*")
    print('****', x0, y0, xg, yg,  th0)

    err_dist = 50
    err = math.sqrt((xg - x0) ** 2 + (yg - y0) ** 2)
    while err > err_dist:
        w = math.atan2(yg -y0, xg -x0) - th0
        xn, yn, thn = diffDriveT(s0, v, w , 0.1, 1)
        err = math.sqrt((xg - x0)**2 + (yg - y0)**2)
        print('err', err, xn[1], yn[1], thn[1])
        plt.plot(xn[1], -yn[1], 'g.')
        s0 = [xn[1], yn[1], thn[1]]
        x0 = xn[1]
        y0 = yn[1]
        th0 = thn[1]
        plt.pause(0.01)
    # plt.plot(rx[subsample*i]*20, ry[subsample*i]*20, 'b.')
plt.show()





