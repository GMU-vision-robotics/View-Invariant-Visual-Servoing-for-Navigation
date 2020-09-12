#!/usr/bin/env python
import argparse
import cv2
import os
import rrt

def main(directory, x0, y0, x1, y1):

    path_finder = rrt.PathFinder(directory)
    print("Loading...")
    path_finder.load()
    print("Finding...")
    solution, lines = path_finder.find(x0, y0, x1, y1)

    if solution:
        print("Writing solution...")
        floormap_file_name = os.path.join(directory, 'floormap.png')
        floormap = cv2.imread(floormap_file_name)

        floormap_with_path_file_name = os.path.join(directory, 'floormap_with_path.png')
        for i in range(len(lines) - 1):
            cv2.line(floormap,
                     (lines[i][1], lines[i][0]),
                     (lines[i+1][1], lines[i+1][0]),
                     (0, 0, 255),
                     10)

        cv2.circle(floormap, tuple([lines[0][1], lines[0][0]]), 50, (0, 255, 0), thickness=30)
        cv2.circle(floormap, tuple([lines[-1][1], lines[-1][0]]), 50, (0, 0, 255), thickness=30)
        cv2.imwrite(floormap_with_path_file_name, floormap)
    else:
        print("No path found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dir', type=str, default='gibson-data/dataset/Allensville')
    parser.add_argument('-x0', type=float, default=-0.591084)
    parser.add_argument('-y0', type=float, default=7.3339)
    parser.add_argument('-x1', type=float, default=5.93709)
    parser.add_argument('-y1', type=float, default=-0.421058)
    args = parser.parse_args()
    main(args.dir, args.x0, args.y0, args.x1, args.y1)
