import os
from time import time

import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop

from .utility import get_image_array, match_color
from .config import d_config, fig_width, fig_length, default_dpi

if not os.path.exists('./output'):
    os.makedirs('./output')

if not os.path.exists('./upload'):
    os.makedirs('./upload')


class Converter:
    def __init__(self, set_id='21226', use_only_colors=None, unlimited_blocks=True, verbose=False):
        if set_id not in d_config:
            raise Exception('set_id is not found in predefined specs. Please manually add it to d_config in config.py.')

        self.verbose = verbose
        self.set_id = set_id
        self.unlimited_blocks = unlimited_blocks

        self.config = d_config[set_id].copy()
        self.config['n_blocks'] = {}
        if use_only_colors is not None:
            assert type(use_only_colors) == list, 'Type error! avoid_colors parameter should be a list.'
            for k in use_only_colors:
                self.config['n_blocks'][k] = d_config[set_id]['n_blocks'][k]
        else:
            self.config['n_blocks'] = d_config[set_id]['n_blocks'].copy()

        self.session_id = ''

        self.image_array_raw = None
        self.image_array_trimmed = None
        self.image_array_filtered = None
        self.image_array_converted = None

    def load_image(self, **kwargs):
        try:
            self.session_id = str(int(time()))
            os.makedirs(f'./output/{self.session_id}')
            a = get_image_array(**kwargs)
            l, w, _ = a.shape
            if l <= self.config['frame_length'] * 10 or w <= self.config['frame_width'] * 10:
                print("Warning: Image quality is low, and Mosaic effect may be nontrivial.")
            self.image_array_raw = a
            if self.verbose:
                print(f"Image loaded successfully. session_id: {self.session_id}")
        except Exception as e:
            print(e)
        return

    def plot_image(self, image_type="raw", save=False, show=False):
        if image_type not in ["raw", "trimmed", "filtered", "converted"]:
            raise Exception("Invalid image_type parameter! Eligible values are raw, trimmed, filtered and converted.")

        if image_type == "raw":
            assert self.image_array_raw is not None, 'No raw image is found. Please load an image first.'
            l, w, _ = self.image_array_raw.shape
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_length / w * l))
            _ = ax.imshow(self.image_array_raw)
            _ = ax.axis('off')

            if save:
                t = str(int(time()))
                fig.savefig(f'./output/{self.session_id}/{t}_raw.png', dpi=default_dpi)
                if self.verbose:
                    print(f"Raw image saved to /output/{self.session_id}/{t}_raw.png")
        elif image_type == "trimmed":
            assert self.image_array_trimmed is not None, 'No trimmed image is found. Please trim an image first.'
            l, w, _ = self.image_array_trimmed.shape
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_length / w * l))
            _ = ax.imshow(self.image_array_trimmed)
            _ = ax.axis('off')
            if save:
                t = str(int(time()))
                fig.savefig(f'./output/{self.session_id}/{t}_trimmed.png', dpi=default_dpi)
                if self.verbose:
                    print(f"Trimmed image saved to /output/{self.session_id}/{t}_trimmed.png")
        elif image_type == "filtered":
            assert self.image_array_filtered is not None, 'No filtered image is found. Please filter an image first.'
            l, w, _ = self.image_array_filtered.shape
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_length / w * l))
            _ = ax.imshow(self.image_array_filtered.astype(np.uint8))
            _ = ax.axis('off')
            if save:
                t = str(int(time()))
                fig.savefig(f'./output/{self.session_id}/{t}_filtered.png', dpi=default_dpi)
                if self.verbose:
                    print(f"Trimmed image saved to /output/{self.session_id}/{t}_filtered.png")
        elif image_type == "converted":
            assert self.image_array_converted is not None, 'No converted image is found. Please convert an image first.'
            l, w, _ = self.image_array_converted.shape
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_length / w * l))
            _ = ax.set_facecolor('k')
            _ = ax.vlines(x=list(range(w+1)), ymin=0, ymax=l, colors='gray', linewidths=.5)
            _ = ax.hlines(y=list(range(l+1)), xmin=0, xmax=w, colors='gray', linewidths=.5)
            for i in range(l):
                for j in range(w):
                    c = plt.Circle((w-j-.5, i+.5), .35, color=tuple(self.image_array_converted[l-i-1, w-j-1, :]/255))
                    _ = ax.add_patch(c)
            _ = ax.set_xlim([0, w+.5])
            _ = ax.set_ylim([-.5, l])
            _ = ax.axis('off')

            if save:
                t = str(int(time()))
                fig.savefig(f'./output/{self.session_id}/{t}_converted.png', dpi=default_dpi)
                if self.verbose:
                    print(f"Converted image saved to /output/{self.session_id}/{t}_converted.png")
        return

    def trim_image(self):
        assert self.image_array_raw is not None, 'No raw image is found! Please load an image first.'
        l, w, _ = self.image_array_raw.shape
        l_, w_ = self.config['frame_length'], self.config['frame_width']

        if l / w > l_ / w_:
            unit = w // w_
            w_new = w_ * unit
            l_new = l_ * unit
        else:
            unit = l // l_
            l_new = l_ * unit
            w_new = w_ * unit

        self.image_array_trimmed = self.image_array_raw[-l_new:, :w_new, :]
        if self.verbose:
            print(f"Raw image is trimmed from {l}×{w} to {l_new}×{w_new}, anchoring at lower left corner.")
        return

    def filter_image(self):
        assert self.image_array_trimmed is not None, "No trimmed image is found! Please trim an image first."
        self.image_array_filtered = np.zeros([
            self.config['frame_length']
            , self.config['frame_width']
            , 3
        ])
        l, w, _ = self.image_array_trimmed.shape
        l_, w_ = self.config['frame_length'], self.config['frame_width']
        unit = l // l_
        assert w // w_ == unit, "Size of trimmed image doesn't match frame size!"

        for i in range(self.config['frame_length']):
            for j in range(self.config['frame_width']):
                self.image_array_filtered[i, j, :] = np.mean(
                    self.image_array_trimmed[unit * i:unit * (i + 1), unit * j:unit * (j + 1), :]
                    , axis=(0, 1)
                ).astype(int)

        if self.verbose:
            print(f"Trimmed image is filtered from {l}×{w} to {l_}×{w_}.")
        return

    def convert_image(self):
        assert self.image_array_filtered is not None, "No filtered image is found! Please filter an image first."
        self.image_array_converted = np.zeros_like(self.image_array_filtered)
        l, w, _ = self.image_array_filtered.shape
        colors = list(self.config['n_blocks'].keys())
        d_image = {(i, j): {'is_matched': False} for i in range(l) for j in range(w)}
        d_blocks = {c: {'n': self.config['n_blocks'][c], 'heap': []} for c in colors}

        for i in range(l):
            for j in range(w):
                res = match_color(
                    tuple((self.image_array_filtered[i, j, :]).astype(int)),
                    colors
                )
                d_image[(i, j)]['candidates'] = res
                for c in res:
                    heappush(d_blocks[c[1]]['heap'], (c[0], (i, j)))

        n_matched_blocks = l * w
        while n_matched_blocks > 0:
            assign_color, min_gap = None, 256 * 3
            for c in colors:
                q, n = d_blocks[c]['heap'], d_blocks[c]['n']
                while q and d_image[q[0][1]]['is_matched'] is True:
                    heappop(q)
                if len(q) > 0 and q[0][0] < min_gap and n > 0:
                    assign_color = c
                    min_gap = q[0][0]
            _, (i, j) = heappop(d_blocks[assign_color]['heap'])
            self.image_array_converted[i, j, :] = assign_color
            if not self.unlimited_blocks:
                d_blocks[assign_color]['n'] -= 1
            d_image[(i, j)]['is_matched'] = True
            d_image[(i, j)]['assigned_color'] = assign_color
            n_matched_blocks -= 1

        if self.verbose:
            print(f"Filtered image has been matched to a set of Lego blocks.")
        return

    def process_image(self):
        self.trim_image()
        self.filter_image()
        self.convert_image()
        return
