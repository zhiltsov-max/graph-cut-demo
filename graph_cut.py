from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import numpy as np
import pymaxflow
import argparse
import os.path as osp


class GraphCut:
    def segment(self, image, fg_points, bg_points):
        eps = 1e-5
        inf = 1e9

        def create_graph(vertices_count, edges_count):
            return pymaxflow.PyGraph(vertices_count, edges_count)
        def add_nodes(graph, count):
            graph.add_node(count)
        def add_edges(graph, src_vertices, dst_vertices, weights, reverse_weights):
            graph.add_edge_vectorized(src_vertices, dst_vertices,
                weights, reverse_weights)
        def set_foreground_weights(graph, points, indices_array, weight=1e10):
            indices = indices_array[points[:, 1], points[:, 0]].ravel()
            graph.add_tweights_vectorized(indices,
                np.zeros(len(points), np.float32),
                np.full(len(points), weight, dtype=np.float32))
        def set_background_weights(graph, points, indices_array, weight=1e10):
            indices = indices_array[points[:, 1], points[:, 0]].ravel()
            graph.add_tweights_vectorized(indices,
                np.full(len(points), weight, dtype=np.float32),
                np.zeros(len(points), np.float32))
        def set_term_weights(graph, indices, weights_up, weights_down):
            graph.add_tweights_vectorized(indices, weights_up, weights_down)
        def compute_adj_dist(v1, v2):
            rsigma2 = 1.0 / 10.0
            return np.exp(- 0.5 * rsigma2 * np.abs(v1 - v2))
        def compute_term_weights(img, fg_points, bg_points):
            fg_hist, fg_bins = np.histogram(img[fg_points[:, 1], fg_points[:, 0]],
                bins=8, range=(0, 256), density=True)
            bg_hist, bg_bins = np.histogram(img[bg_points[:, 1], bg_points[:, 0]],
                bins=8, range=(0, 256), density=True)

            fg_ind = np.searchsorted(fg_bins[1:-1], img.flatten())
            bg_ind = np.searchsorted(bg_bins[1:-1], img.flatten())

            fg_prob = fg_hist[fg_ind] + eps
            bg_prob = bg_hist[bg_ind] + eps

            scale = 1.0
            fg_weights = -scale * np.log(fg_prob)
            bg_weights = -scale * np.log(bg_prob)
            return fg_weights, bg_weights

        image = image.convert('L')
        im = np.array(image)

        indices = np.arange(im.size, dtype=np.int32).reshape(im.shape)
        fg_points = np.array(fg_points, dtype=int)
        bg_points = np.array(bg_points, dtype=int)

        g = create_graph(im.size, im.size * 4)
        add_nodes(g, im.size)

        # adjacent down
        diffs = (compute_adj_dist(im[:, 1:], im[:, :-1]) + eps) \
            .astype(np.float32).ravel()
        e1 = indices[:,  :-1].ravel()
        e2 = indices[:, 1:  ].ravel()
        add_edges(g, e1, e2, diffs, 0 * diffs)

        # adjacent up
        diffs = (compute_adj_dist(im[:, :-1], im[:, 1:]) + eps) \
            .astype(np.float32).ravel()
        e1 = indices[:, :-1].ravel()
        e2 = indices[:, 1:].ravel()
        add_edges(g, e1, e2, diffs, 0 * diffs)

        # adjacent right
        diffs = (compute_adj_dist(im[1:, 1:], im[:-1, :-1]) + eps) \
            .astype(np.float32).ravel()
        e1 = indices[1:,    :-1].ravel()
        e2 = indices[ :-1, 1:  ].ravel()
        add_edges(g, e1, e2, diffs, 0 * diffs)

        # adjacent left
        diffs = (compute_adj_dist(im[:-1, :-1], im[1:, 1:]) + eps) \
            .astype(np.float32).ravel()
        e1 = indices[:-1, :-1].ravel()
        e2 = indices[1:, 1:].ravel()
        add_edges(g, e1, e2, diffs, 0 * diffs)

        fg_weights, bg_weights = compute_term_weights(im, fg_points, bg_points)
        set_term_weights(g, indices.ravel(),
            fg_weights.astype(np.float32).ravel(),
            bg_weights.astype(np.float32).ravel())

        # links the to source and sink
        set_foreground_weights(g, fg_points, indices, inf)
        set_background_weights(g, bg_points, indices, inf)

        g.maxflow()

        out = g.what_segment_vectorized()
        return out.reshape(im.shape)

class View:
    def __init__(self, args):
        self.window = Tk()
        self.window.title("Image segmentation demo")
        self.main_frame = ttk.Frame(self.window, padding="3 3 12 12")
        self.main_frame.grid(column=0, row=0, sticky=(N, W, E, S))
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        self.clear_btn = ttk.Button(self.main_frame, text="Clear")
        self.clear_btn['command'] = lambda: self.clear_button_handler()
        self.clear_btn.grid(column=0, row=0, sticky=(W, N))
        self.segm_btn = ttk.Button(self.main_frame, text="Segment")
        self.segm_btn['command'] = lambda: self.segment_button_handler()
        self.segm_btn.grid(column=1, row=0, sticky=(W, N))
        self.src_image = Image.open(osp.abspath(args.image))
        self.src_img = ImageTk.PhotoImage(self.src_image)
        self.src_cnv = Canvas(self.main_frame,
            width=self.src_img.width(), height=self.src_img.height())
        self.src_cnv.grid(column=0, row=1, sticky=(W, N))
        self.src_cnv.bind("<B1-Motion>", lambda e: self.fg_drawing_handler(e))
        self.src_cnv.bind("<B3-Motion>", lambda e: self.bg_drawing_handler(e))
        self.dst_image = None
        self.dst_img = None
        self.dst_cnv = Canvas(self.main_frame,
            width=self.src_img.width(), height=self.src_img.height())
        self.dst_cnv.grid(column=1, row=1, sticky=(E, N))
        for child in self.main_frame.winfo_children(): child.grid_configure(padx=5, pady=5)

        self.clear_button_handler()
        self.graph_cut = GraphCut()

    def run(self):
        self.window.mainloop()

    def clear_button_handler(self):
        self.src_cnv.delete("all")
        self.dst_cnv.delete("all")
        self.fg_points = []
        self.bg_points = []
        self.dst_image = None

        self.src_cnv.create_image(
            (self.src_img.width() // 2, self.src_img.height() // 2),
            image=self.src_img)

    def segment_button_handler(self):
        self.dst_image = self.segment()
        self.dst_img = ImageTk.PhotoImage(self.dst_image)
        self.dst_cnv.create_image(
            (self.src_img.width() // 2, self.src_img.height() // 2),
            image=self.dst_img)

    def clip_point(self, x, y):
        w, h = self.src_image.size
        return ( max(min(w - 1, x), 0), max(min(h - 1, y), 0) )

    def fg_drawing_handler(self, event):
        x1, y1 = ( event.x - 1 ), ( event.y - 1 )
        x2, y2 = ( event.x + 1 ), ( event.y + 1 )
        self.fg_points.append(self.clip_point(event.x, event.y))
        self.src_cnv.create_oval(x1, y1, x2, y2,
            width=2, outline="#ff0000")

    def bg_drawing_handler(self, event):
        x1, y1 = ( event.x - 1 ), ( event.y - 1 )
        x2, y2 = ( event.x + 1 ), ( event.y + 1 )
        self.bg_points.append(self.clip_point(event.x, event.y))
        self.src_cnv.create_oval(x1, y1, x2, y2,
            width=2, outline="#00ff00")

    def segment(self):
        mask = self.graph_cut.segment(self.src_image,
            self.fg_points, self.bg_points)
        mask = (mask * 255).astype(np.uint8)
        image = Image.fromarray(mask, mode='L')
        return image

def parse_args(source=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("image", metavar="PATH",
        help="Path to the image for segmentation")

    return parser.parse_args(source)

def main():
    args = parse_args()

    view = View(args)
    view.run()

if __name__ == "__main__":
    main()