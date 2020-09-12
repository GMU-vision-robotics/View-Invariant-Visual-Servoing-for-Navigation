#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt
import meshcut
import numpy as np
import pdb

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

def main(mesh_file_name):
    verts, faces = load_obj(mesh_file_name)
    z =  np.min(verts[:,-1]) + 0.5 # 0.5 is the height of husky, actually it should be 0.37
    # cut the mesh with a surface whose value on z-axis is plane_orig, and its normal is plane_normal vector
    #print('verts: {}'.format(verts[0]))
    print('meshcut ...')
    cross_section = meshcut.cross_section(verts, faces, plane_orig=(0, 0, z), plane_normal=(0, 0, 1))
    n_nodes = sum([len(x) for x in cross_section])
    n_edges = n_nodes - len(cross_section)
    output_file_name = mesh_file_name.replace('.obj', '_2d.obj')
    with open(output_file_name, 'w') as output_file:
        output_file.write("# {} vertices, {} edges\n".format(n_nodes, n_edges))
        for item in cross_section:
            for node in item:
                output_file.write("v {:0.6f} {:0.6f}\n".format(node[0], node[1]))

        i = 0
        for item in cross_section:
            print("{}".format(len(item)))
            for _ in range(len(item) - 1):
                output_file.write("e {} {}\n".format(i, i+1))
                i+=1
            i+=1

if __name__ == "__main__":    
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mesh_name', type=str, default='gibson-data/dataset/Allensville/mesh_z_up.obj')
    args = parser.parse_args()
    main(args.mesh_name)

