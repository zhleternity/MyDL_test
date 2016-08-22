#!/usr/bin/env python
# _*_ coding: utf-8 _*_

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
    
