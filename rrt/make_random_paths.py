#!/usr/bin/env python
import argparse
import cProfile
from collections import defaultdict, namedtuple
import cv2
import find_path
import heapq
import math
import multiprocessing as mp
import numpy as np
import os
import pdb
import pickle
import random
import rrt

def main(directory, n):

    path_finder = find_path.PathFinder(directory)
    path_finder.load()
    pc = path_finder.pc
    floormap_file_name = os.path.join(directory, 'floormap.png')
    floormap  = cv2.imread(floormap_file_name)

    to_do = n
    iteration = 0
    args = []
    
    while to_do > iteration:
        p0 = pc.random_free_point()
        p1 = pc.random_free_point()
        if p0 != p1:
            iteration+=1
            args.append((p0[0], p0[1], p1[0], p1[1]))
    
    iteration = 0
    pool = mp.Pool()
    node_count = defaultdict(int)
    edge_count = defaultdict(int)

    for solution, lines in pool.starmap(path_finder.find, args):
        iteration += 1
        if (iteration+1)%10 == 0:
            print("{}/{}".format(iteration+1,n))

        if solution:
            last_node = None
            for node in solution.path:
                if last_node:
                    edge_count[(last_node, node)] += 1
                node_count[node] += 1
                last_node = node
            for i in range(len(lines) - 1):
                cv2.line(floormap, (lines[i][1], lines[i][0]), (lines[i+1][1], lines[i+1][0]), (0,0,255), 10)

#            cv2.circle(floormap, tuple([lines[0][1], lines[0][0]]),  50, (0, 255, 0), thickness=30)
#            cv2.circle(floormap, tuple([lines[-1][1], lines[-1][0]]),50, (0, 0, 255), thickness=30)
        else:
            print("No path found from {} to {}".format(p0, p1))

    target_width = 1800
    height, width, depth = floormap.shape
    image_scale = target_width/width
    new_x,new_y = floormap.shape[1]*image_scale, floormap.shape[0]*image_scale
    new_floormap = cv2.resize(floormap,(int(new_x),int(new_y)))
    cv2.imshow('Floor map', new_floormap)
    while True:
        # wait for "q"
        key = cv2.waitKey(0)
        if key == 113:
            cv2.destroyAllWindows()
            break

    pdb.set_trace()
    return

if __name__ == "__main__":    
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dir', type=str, default='gibson-data/dataset/Allensville')
    parser.add_argument('-n', type=int, default=100)
    parser.add_argument('-p', type=str, default='', help='Run cProfile on main() function and store results in file provided.')
    args = parser.parse_args()
    if args.p:
        cProfile.run("""main(args.dir, args.n)""", args.p)
    else:
        main(args.dir, args.n)
    
