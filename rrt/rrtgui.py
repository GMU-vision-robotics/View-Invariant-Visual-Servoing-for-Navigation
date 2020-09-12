#!/usr/bin/env python
import gi                           # pylint: disable=wrong-import-order,wrong-import-position
gi.require_version("Gtk", "3.0")    # pylint: disable=wrong-import-order,wrong-import-position
from collections import defaultdict
from gi.repository import Gtk, GLib, Gdk # pylint: disable=wrong-import-order,wrong-import-position
import math
import numpy as np
import os
import pdb
import random
import sys
import threading
import time
import traceback
import cairo                        # pylint: disable=wrong-import-order,wrong-import-position
import rrt

def filled_circle(ctx, center, radius, color):
    ctx.set_source_rgb(*color)
    ctx.arc(
        center[0],
        center[1],
        radius,
        0,
        2*math.pi)
    ctx.fill()

class RrtDisplay(Gtk.Window):
    def __init__(self):

        Gtk.Window.__init__(self, title="RRT Planner")
        self.set_default_size(500, 500)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.add(vbox)
        box = Gtk.Box(spacing=6)
        vbox.pack_start(box, False, True, 0)

        self.load_button = Gtk.Button("Load")
        self.load_button.connect("clicked", self.on_folder_clicked)
        box.pack_start(self.load_button, True, True, 0)

        self.rrt_button = Gtk.Button("Make RRT")
        self.rrt_button.connect("clicked", self.on_make_rrt_clicked)
        self.rrt_button.set_sensitive(False)
        box.pack_start(self.rrt_button, True, True, 0)

        self.find_button = Gtk.Button("Find Path")
        self.find_button.connect("clicked", self.on_find_path_clicked)
        self.find_button.set_sensitive(False)
        box.pack_start(self.find_button, True, True, 0)

        self.refine_button = Gtk.Button("Refine Path")
        self.refine_button.connect("clicked", self.on_refine_path_clicked)
        self.refine_button.set_sensitive(False)
        box.pack_start(self.refine_button, True, True, 0)

        self.multi_refine_button = Gtk.Button("Multi Refine")
        self.multi_refine_button.connect("clicked", self.on_multi_refine_path_clicked)
        self.multi_refine_button.set_sensitive(True)
        box.pack_start(self.multi_refine_button, True, True, 0)

        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.connect('draw', self.draw)
        self.drawing_area.set_events(
            self.drawing_area.get_events()
            | Gdk.EventMask.BUTTON_PRESS_MASK
            | Gdk.EventMask.BUTTON_RELEASE_MASK)

        self.connect("key-press-event", self.key_pressed)
        self.connect("key-release-event", self.key_release)
        self.drawing_area.connect('button-press-event', self.mouse_click)
        self.drawing_area.connect('button-release-event', self.mouse_release)

        vbox.pack_start(self.drawing_area, True, True, 0)

        self.status_bar = Gtk.Statusbar()
        self.context_id = self.status_bar.get_context_id("rrt")
        vbox.pack_start(self.status_bar, False, True, 0)
        self.status_bar.push(self.context_id, "Pick a Gibson folder to load")

        self.scale = 0.08
        self.epsilon = 0.15
        self.erosion_iterations = 5
        self.num_nodes = 12000
#        self.num_nodes = 5000
        self.z = 8.0
#        self.z = 0.5
        self.pf = None
        self.solution = None
        self.cross_section_2d = None
        self.floor_map = None
        self.free = None
        self.object_file_name = None

        self.pressed = None
        self.path_start_point = None
        self.path_end_point = None
        self.path_start_px = None
        self.path_end_px = None
        self.node_x = None
        self.node_y = None
        self.obj_file_name = None

    def key_pressed(self, widget, event, data=None): # pylint: disable=unused-argument
        key = Gdk.keyval_name(event.keyval)
        self.pressed = key

    def key_release(self, widget, event, data=None): # pylint: disable=unused-argument
        self.pressed = None

    def mouse_click(self, widget, event): # pylint: disable=unused-argument
        if self.pf and self.pf.pc:
            pixels = event.x, event.y
            map_point = self.pf.pc.pixel_to_point(
                (event.x/self.scale,
                 event.y/self.scale))
            if self.pressed == 'Shift_L':
                self.path_end_point = map_point
                self.path_end_px = pixels
            else:
                self.path_start_point = map_point
                self.path_start_px = pixels
            self.drawing_area.queue_draw()
            if self.path_start_point and self.path_end_point:
                GLib.idle_add(self.find_button.set_sensitive, (True,))

    def mouse_release(self, widget, event): # pylint: disable=unused-argument
        pass

    def load_object_file_worker(self, object_file_name):
        px_per_meter = 500
        padding_meters = 0.5
        erosion_iterations = 5

        self.send_status_message("Loading object file")
        verts, faces = rrt.load_obj(object_file_name)
        self.send_status_message("Finding 2D cross-section")
        cross_section = rrt.get_cross_section(verts, faces, z=self.z)
        cross_section_2d = [c[:, 0:2] for c in cross_section]

        self.send_status_message("Creating free mask")
        _, free = rrt.make_free_space_image(
            cross_section_2d,
            px_per_meter,
            padding_meters,
            erosion_iterations=erosion_iterations)

        min_x, max_x, min_y, max_y = rrt.cross_section_bounds(cross_section_2d, padding_meters)

        self.pf = rrt.PathFinder(
            os.path.dirname(object_file_name),
            free=free,
            pc=rrt.PointConverter(
                (min_x, max_x, min_y, max_y),
                px_per_meter,
                padding_meters, free),
            cross_section_2d=cross_section_2d,
            )
        self.object_file_name = object_file_name
        self.object_file_loaded()

    def object_file_loaded(self):
        GLib.idle_add(self.load_button.set_sensitive, (True,))
        GLib.idle_add(self.rrt_button.set_sensitive, (True,))
        self.send_status_message("Loaded {}".format(os.path.basename(self.object_file_name)))
        self.drawing_area.queue_draw()

    def load_object_file(self, obj_file_name):
        self.pf = None
        self.solution = None
        self.cross_section_2d = None
        self.floor_map = None
        self.free = None
        self.obj_file_name = None
        GLib.idle_add(self.rrt_button.set_sensitive, (False,))
        self.drawing_area.queue_draw()
        GLib.idle_add(self.rrt_button.set_sensitive, (False,))
        GLib.idle_add(self.load_button.set_sensitive, (False,))

        thread = threading.Thread(target=self.load_object_file_worker, args=(obj_file_name,))
        thread.daemon = True
        thread.start()

    def find_path_worker(self):
        solution, _ = self.pf.find(
            self.path_start_point[0],
            self.path_start_point[1],
            self.path_end_point[0],
            self.path_end_point[1])
        self.solution = solution
        self.send_redraw()
        GLib.idle_add(self.refine_button.set_sensitive, (True,))

    def on_find_path_clicked(self, widget): # pylint: disable=unused-argument
        if not (self.path_start_point and self.path_end_point):
            GLib.idle_add(self.send_status_message,
                          "Both start and end points must be defined to find a path.")
            return

        thread = threading.Thread(target=self.find_path_worker)
        thread.daemon = True
        thread.start()

    def refine_path_worker(self,):
        path_nodes = self.solution.path
        node_map = [node for node in self.solution.path]

        num_nodes = len(path_nodes)
        nodes_x = np.array([self.pf.nodes_x[node] for node in path_nodes])
        nodes_y = np.array([self.pf.nodes_y[node] for node in path_nodes])
        points = [self.pf.pc.point_to_pixel((nodes_x[i], nodes_y[i])) for i in range(num_nodes)]
        last_cost = self.solution.cost

        edges = defaultdict(dict)
        for idx, node in enumerate(path_nodes):
            if idx:
                node0, node1 = idx-1, idx
                distance = math.sqrt((nodes_x[node0] - nodes_x[node1])**2 +
                                     (nodes_y[node0] - nodes_y[node1])**2)
                edges[node0][node1] = distance
                edges[node1][node0] = distance

        solution = None
        last_solution = None
        max_edge_tries = num_nodes
        max_iters = 1000
        tolerance = 0.0000001
        max_zero_diffs = 100
        n_zero_diffs = 0

        iters = 0
        mesg = ""
#        pdb.set_trace()
        while True:
            iters += 1
            node0 = None
            node1 = None
            found = False
            for i in range(max_edge_tries):
                # if not last_solution:
                #     nodes = list(range(num_nodes))
                # else:
                #     nodes = last_solution.path
                nodes = list(range(num_nodes))
                node0 = nodes[int(random.random() * len(nodes))]
                node1 = nodes[int(random.random() * len(nodes))]

                if(node0 != node1 and
                   node0 not in edges[node1] and
                   rrt.line_check(points[node0], points[node1], self.pf.free, skip=5)):
                    found = True
                    break
            if found:
                distance = math.sqrt((nodes_x[node0] - nodes_x[node1])**2 +
                                     (nodes_y[node0] - nodes_y[node1])**2)
                edges[node0][node1] = distance
                edges[node1][node0] = distance

                # self.pf.edges[node_map[node0]][node_map[node1]] = distance
                # self.pf.edges[node_map[node1]][node_map[node0]] = distance
                # n_edges = self.pf.edges_idx.shape[0]
                # edges_idx = np.zeros((n_edges + 1, 2), dtype=np.int64)
                # edges_idx[:n_edges,:] = self.pf.edges_idx
                # edges_idx[n_edges] = (node_map[node0], node_map[node1])
                # self.pf.edges_idx = edges_idx
                # self.send_redraw()
                
                pf = rrt.PathFinder(
                    free=self.pf.free,
                    pc=self.pf.pc,
                    nodes_x=nodes_x,
                    nodes_y=nodes_y,
                    edges=edges
                )
                solution, _ = pf.find(
                    self.path_start_point[0],
                    self.path_start_point[1],
                    self.path_end_point[0],
                    self.path_end_point[1])
                if not solution:
                    mesg = "Didn't get back a path solution"
                    solution = last_soluton or None
                    break
                cost = solution.cost
                delta = last_cost - cost
                if delta < 0:
                    last_cost = cost
                    last_solution = solution
                    continue
                if delta < tolerance:
                    mesg = "Got a diff {} < which is less than {}".format(delta, tolerance)
                    n_zero_diffs += 1
                    if n_zero_diffs == max_zero_diffs:
                        break
                    else:
                        continue
                n_zero_diffs = 0
                last_cost = cost
                last_solution = solution
                if iters > max_iters:
                    mesg = "Ran out of iterations {}".format(max_iters)
                    break
            else:
                mesg = "Couldn't find a new edge after {} tries.".format(max_edge_tries)
                break

        self.send_status_message(mesg)
        
        if solution:
            # Add new edges in path to RRT
            edges_to_add = []
            for i, node1 in enumerate(solution.path):
                if i:
                    node0 = solution.path[i-1]
                    if node_map[node1] not in self.pf.edges[node_map[node0]]:
                        edges_to_add.append((node0, node1))
            for node0, node1 in edges_to_add:
                self.pf.edges[node_map[node0]][node_map[node1]] = edges[node0][node1]
                self.pf.edges[node_map[node1]][node_map[node0]] = edges[node1][node0]
            self.pf.edges_idx = np.vstack(
                (self.pf.edges_idx,
                 np.array([(node_map[i],node_map[j]) for i,j in edges_to_add],
                          dtype=np.int64)))
                        
            # Copy to path to main path
            for i, node in enumerate(solution.path):
                solution.path[i] = node_map[node]
            self.solution = solution
            self.send_redraw()

    def on_refine_path_clicked(self, widget): # pylint: disable=unused-argument
        thread = threading.Thread(target=self.refine_path_worker)
        thread.daemon = True
        thread.start()

    def on_multi_refine_path_clicked(self, widget): # pylint: disable=unused-argument
        thread = threading.Thread(target=self.refine_multi_path_worker)
        thread.daemon = True
        thread.start()

    def refine_multi_path_worker(self):
        for _ in range(100):
            start_point = self.pf.pc.random_free_point()
            finish_point = self.pf.pc.random_free_point()

            self.path_start_point = start_point
            self.path_start_px = self.pf.pc.point_to_pixel(start_point)
            self.path_end_point = finish_point
            self.path_end_px = self.pf.pc.point_to_pixel(finish_point)
            
            solution, _ = self.pf.find(
                self.path_start_point[0],
                self.path_start_point[1],
                self.path_end_point[0],
                self.path_end_point[1])
            if solution:
                self.solution = solution
                self.refine_path_worker()
        self.solution = None
        self.path_start_point = None
        self.path_start_px = None
        self.path_end_point = None
        self.path_end_px = None
        self.send_redraw()

    def on_folder_clicked(self, widget): # pylint: disable=unused-argument
        dialog = Gtk.FileChooserDialog(
            "Choose a folder", self,
            Gtk.FileChooserAction.SELECT_FOLDER,
            (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
             "Select", Gtk.ResponseType.OK))
        dialog.set_default_size(800, 400)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            folder = dialog.get_filename()
            obj_file_name = os.path.join(folder, 'mesh_z_up.obj')
            if os.path.exists(obj_file_name):
                self.load_object_file(obj_file_name)
            else:
                err_file_name = '.../' + '/'.join(obj_file_name.split('/')[-2:])
                GLib.idle_add(
                    self.send_status_message,
                    "File {} does not exist. Try another folder.".format(err_file_name))
        dialog.destroy()

    def make_rrt_worker(self):
        counter = 0
        def callback(nodes_x, nodes_y, edges):
            nonlocal counter
            counter += 1
            if counter%100 == 0:
                self.pf.nodes_x = nodes_x
                self.pf.nodes_y = nodes_y
                self.pf.edges_idx = np.copy(edges)
                self.send_redraw()

        self.send_status_message("Creating RRT")
        _, _, self.pf.nodes_x, self.pf.nodes_y, self.pf.edges_idx = rrt.make_rrt(
            self.pf.cross_section_2d,
            self.pf.pc.padding_meters,
            self.pf.pc.px_per_meter,
            self.num_nodes,
            self.epsilon,
            self.pf.free,
            new_edge_callback=callback,
            crossing_paths=False,
            )

        edges = defaultdict(dict)
        edges_idx = self.pf.edges_idx
        nodes_x = self.pf.nodes_x
        nodes_y = self.pf.nodes_y
        for edge_idx in edges_idx:
            n0 = edge_idx[0]
            n1 = edge_idx[1]
            distance = math.sqrt((nodes_x[n0] - nodes_x[n1])**2 +
                                 (nodes_y[n0] - nodes_y[n1])**2)
            edges[n0][n1] = distance
            edges[n1][n0] = distance
        self.pf.edges = edges
        self.rrt_made()

    def rrt_made(self):
        GLib.idle_add(self.rrt_button.set_sensitive, (True,))
        self.send_status_message("Created RRT")
        self.drawing_area.queue_draw()

    def clear_start_and_end(self):
        self.path_start_point = None
        self.path_end_point = None
        self.path_start_px = None
        self.path_end_px = None

    def clear_path(self):
        self.clear_start_and_end()
        self.solution = None

    def make_rrt(self):
        self.clear_path()
        GLib.idle_add(self.load_button.set_sensitive, (False,))
        GLib.idle_add(self.rrt_button.set_sensitive, (False,))
        thread = threading.Thread(target=self.make_rrt_worker,)
        thread.daemon = True
        thread.start()

    def on_make_rrt_clicked(self, widget): # pylint: disable=unused-argument
        self.make_rrt()

    def send_status_message(self, message):
        self.status_bar.push(self.context_id, message)

    def send_redraw(self):
        self.drawing_area.queue_draw()

    def draw_map(self, ctx):
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(1)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND) # pylint: disable=no-member

        p2p = self.pf.pc.scaled_point_to_pixel

        for line in self.pf.cross_section_2d:
            line = [p2p((x[0], x[1]), self.scale) for x in line]
            ctx.new_path()
            ctx.move_to(line[0][0], line[0][1])
            for p in line[1:]:
                ctx.line_to(p[0], p[1])
            ctx.stroke()

    def draw_rrt(self, ctx):
        node_x = self.node_x
        node_y = self.node_y
        ctx.set_source_rgb(0, 0, 1.0)
        for src, dst in self.pf.edges_idx:
            ctx.new_path()
            ctx.move_to(node_x[src], node_y[src])
            ctx.line_to(node_x[dst], node_y[dst])
            ctx.stroke()

    def draw_solution(self, ctx):
        ctx.set_source_rgb(1.0, 0, 0)
        ctx.set_line_width(5)
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

    def draw(self, da, ctx): # pylint: disable=arguments-differ
        ctx.set_source_rgb(1, 1, 1)
        ctx.rectangle(0, 0, da.get_allocated_width(), da.get_allocated_height())
        ctx.fill()

        ctx.save()
        if self.pf:
            self.draw_map(ctx)

            if self.pf.nodes_x is not None:
                x2px = self.pf.pc.x_to_pixel
                y2px = self.pf.pc.y_to_pixel
                self.node_x = [x2px(x)*self.scale for x in self.pf.nodes_x]
                self.node_y = [y2px(y)*self.scale for y in self.pf.nodes_y]
                self.draw_rrt(ctx)

                if self.solution:
                    self.draw_solution(ctx)
                if self.path_start_px:
                    filled_circle(ctx, self.path_start_px, 10, (0, 1, 0))
                if self.path_end_px:
                    filled_circle(ctx, self.path_end_px, 10, (1, 0, 0))
        ctx.restore()

if __name__ == '__main__':
    try:
        win = RrtDisplay()
        win.connect("destroy", Gtk.main_quit)
        win.show_all()
        Gtk.main()
    except Exception as e:
        print("Exception {}: {}".format(e, traceback.format_exception(*sys.exc_info())))
        raise
