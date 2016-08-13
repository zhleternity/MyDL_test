import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
    """
    given an image pixels info,and the function of extarcting features,using them we extract features on the image set,
    and save as a feature array
    Inputs:
    :param imgs: N x H X W x C array of pixel data for N images
    :param feature_fns: List of k feature function
    :param verbose:
    :return:
    """
