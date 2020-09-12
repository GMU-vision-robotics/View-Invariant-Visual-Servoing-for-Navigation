#!/usr/bin/env python
import cv2
import numpy as np
import os
import pickle

import rrt

def main(mesh_file_name, px_per_meter,
         padding_meters, num_nodes,
         epsilon, extend_lines,
         crossing_paths, erosion_iterations,z):

    verts, faces = rrt.load_obj(mesh_file_name)
    # cross_section is a list of lines
    cross_section = rrt.get_cross_section(verts, faces, z)
    #print(cross_section.shape)


    cross_section_2d = [c[:, 0:2] for c in cross_section]

    floor_map, free = rrt.make_free_space_image(cross_section_2d, px_per_meter,
                                                padding_meters,
                                                erosion_iterations=erosion_iterations)
    print('finished making free space image.')
    edges_from_px, edges_to_px, nodes_x, nodes_y, edges = rrt.make_rrt(cross_section_2d,
                                                                       padding_meters,
                                                                       px_per_meter,
                                                                       num_nodes,
                                                                       epsilon,
                                                                       free,
                                                                       extend_lines,
                                                                       crossing_paths,
                                                                       )
    print('floormap.shape: {}'.format(floor_map.shape))
    print('free.shape: {}'.format(free.shape))
#'''
    floor_map_white = (floor_map == 255)
    floor_map_black = (floor_map == 0)
    floor_map[floor_map_white] = 0
    floor_map[floor_map_black] = 255

    for i, edge_from_px in enumerate(edges_from_px):
        cv2.line(floor_map, edge_from_px, edges_to_px[i], (255, 0, 0), thickness=5)

    min_x, max_x, min_y, max_y = rrt.cross_section_bounds(cross_section_2d, padding_meters)

    mesh_dir = os.path.dirname(mesh_file_name)
    free_file_name = os.path.join(mesh_dir, 'free.png')
    floormap_file_name = os.path.join(mesh_dir, 'floormap.png')
    rrt_file_name = os.path.join(mesh_dir, 'rrt.npz')
    config_file_name = os.path.join(mesh_dir, 'config.pickle')

    ## the floormap image, with white background and black boundaries
    cv2.imwrite(floormap_file_name, floor_map)
    print("Wrote {}".format(floormap_file_name))
    np.savez(rrt_file_name, nodes_x, nodes_y, edges, free, *cross_section_2d)
    print("Wrote {}".format(rrt_file_name))
    with open(config_file_name, 'wb') as config_file:
        pickle.dump({
            'px_per_meter': px_per_meter,
            'padding_meters': padding_meters,
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            }, config_file)
    print("Wrote {}".format(config_file_name))
    ## the free image, with black background as occupancy and white foreground as free space
    cv2.imwrite(free_file_name, free)
    print("Wrote {}".format(free_file_name))
#'''

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mesh_name', type=str,
                        default='/home/reza/Datasets/GibsonEnv/gibson/assets/dataset/Allensville_for_rrt/mesh_z_up.obj')
                        #'gibson-data/dataset/Allensville/mesh_z_up.obj')
    parser.add_argument('-n', '--num_nodes', type=int, default=5000)
    parser.add_argument('-i', '--erosion_iterations', type=int, default=5)
    parser.add_argument('-e', '--epsilon', type=float, default=.1)
    parser.add_argument('-z', '--height', type=float, default=3.367)
    parser.add_argument('-l', '--extend_lines', default=False, action="store_true")
    parser.add_argument('-x', '--no_crossing_paths', default=False, action="store_true")
    args = parser.parse_args()
    main(args.mesh_name, 500,
         0.5, args.num_nodes,
         args.epsilon, args.extend_lines,
         not args.no_crossing_paths, args.erosion_iterations,z=args.height)
