#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# __author__ = 'eternity'

from math import sqrt, ceil
import numpy as np

def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4d tensor of image data to a grid for easy visualization.
    :param Xs: Data of shape (N, H, W, C)
    :param ubound: Output grid will have values scaled to the range [0, ubound]
    :param padding: The number of blank pixels between elements of the grid
    :return:
    """

    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in xrange(grid_size):
        x0, x1 = 0, W
        for x in xrange(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid

def visual_grid(Xs):
    """visualize a grid of images"""
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A*H+A, A*W+A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0
    for y in xrange(A):
        for x in xrange(A):
             if n < N:
                 G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = Xs[n, :, :, :]
                 n += 1
    #  normalize to [0,1]
    max_g = G.max()
    min_g = G.min()
    G = (G - min_g) / (max_g - min_g)
    return G

def visual_nn(rows):
    """visualize array of arrays of images"""
    N = len(rows)
    D = len(rows[0])
    H, W, C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N*H+N, D*W+D, C),Xs.dtype)
    for y in xrange(N):
        for x in xrange(D):
            G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = rows[y][x]
    max_g = G.max()
    min_g = G.min()
    G = (G - min_g) / (max_g - min_g)
    return G




































































