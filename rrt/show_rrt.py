#!/usr/bin/env python
import argparse
import cairo
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
import numpy as np
import os
import pdb

import rrt

class RrtDisplay:
    def __init__(self,
                 drawing_area,
                 path_finder,
                 width,
                 height,
                 scale,
                 solution=None
                ):
        self.da = drawing_area
        self.pf = path_finder
        self.width = width
        self.height = height
        self.scale = scale
        self.solution = solution

    def draw_map(self, da, ctx):
        ctx.set_source_rgb(1,1,1)
        # width, height = ctx.get_size()
        ctx.rectangle(0,0,da.get_allocated_width(), da.get_allocated_height())
#        ctx.rectangle(0, 0, self.width, self.height)
        ctx.fill()
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(1)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)

        p2p = self.pf.pc.scaled_point_to_pixel

        for idx, line in enumerate(self.pf.cross_section_2d):
            line = [p2p((x[0], x[1]), self.scale) for x in line]
            ctx.new_path()
            ctx.move_to(line[0][0], line[0][1])
            for p in line[1:]:
                ctx.line_to(p[0], p[1])
            ctx.stroke()

    def draw_rrt(self, da, ctx):
        node_x = self.node_x
        node_y = self.node_y
        ctx.set_source_rgb(0, 0, 1.0)
        for src, dst in self.pf.edges_idx:
            ctx.new_path()
            ctx.move_to(node_x[src], node_y[src])
            ctx.line_to(node_x[dst], node_y[dst])
            ctx.stroke()

    def draw_solution(self, da, ctx):
        ctx.set_source_rgb(1.0, 0, 0)
        px_path = []
        node_x = self.node_x
        node_y = self.node_y
        for node in self.solution.path:
            p = (node_x[node], node_y[node])
            px_path.append(p)
        ctx.new_path()
        ctx.move_to(px_path[0][0], px_path[0][1])
        for p in px_path[1:]:
            ctx.line_to(p[0], p[1])
        ctx.stroke()

    def draw(self, da, ctx):
        x2px = self.pf.pc.x_to_pixel
        y2px = self.pf.pc.y_to_pixel
        self.node_x = node_x = [x2px(x)*self.scale for x in self.pf.nodes_x]
        self.node_y = node_y = [y2px(y)*self.scale for y in self.pf.nodes_y]

        ctx.save()
        self.draw_map(da, ctx)
        self.draw_rrt(da,ctx)

        if self.solution:
            self.draw_solution(da, ctx)
        ctx.restore()
        return 

def main(directory, scale, x0, y0, x1, y1):
    pf = rrt.PathFinder(directory)
    pf.load()
    bounds = pf.get_bounds()
    solution, path_lines = pf.find(x0, y0, x1, y1)

    width = pf.pc.x_to_pixel(bounds.max_x - bounds.min_x)*scale
    height = pf.pc.x_to_pixel(bounds.max_y - bounds.min_y)*scale

    win = Gtk.Window()
    win.connect('destroy', Gtk.main_quit)
    win.set_default_size(width, height)

    drawing_area = Gtk.DrawingArea()

    rrt_display = RrtDisplay(
        drawing_area,
        pf,
        width,
        height,
        scale,
        solution
        )

    win.add(drawing_area)
    drawing_area.connect('draw', rrt_display.draw)
    win.show_all()
    Gtk.main()

if __name__ == "__main__":    
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dir', type=str, default='gibson-data/dataset/Allensville')
    parser.add_argument('-s', '--scale', type=float, default=0.1)
    parser.add_argument('-x0', type=float, default=-0.591084)
    parser.add_argument('-y0', type=float, default=7.3339)
    parser.add_argument('-x1', type=float, default=5.93709)
    parser.add_argument('-y1', type=float, default=-0.421058)
    args = parser.parse_args()
    main(args.dir, args.scale, args.x0, args.y0, args.x1, args.y1)
    
