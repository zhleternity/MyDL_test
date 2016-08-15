import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
    """
    given an image pixels info,and the function of extarcting features,using them we extract features on the image set,
    and save as a feature array
    Inputs:
    :param imgs: N x H X W x C array of pixel data for N images
    :param feature_fns: List of k feature functions,the ith feature function should take
    as input an H x W x D array and return a (one-dimensional)array of length F_i.
    :param verbose:if true,print progress
    :return:
    an array of shape(F_1 + ... + F_k, N) where each column is the concatenation of all features for a single image.
    """

    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])
    #use the first image to determine feature dimensions
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
        feature_dims.append(feats.size)
        first_image_features.append(feats)
    #now that we know the dimensions of the features, we can allocate a single
    #big array to store all features as columns
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros(total_feature_dim, num_images)
    imgs_features[:total_feature_dim, 0] = np.hstack(first_image_features)

    #extract features for the rest of the images
    for i in xrange(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[idx:next_idx, i] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 0:
            print 'Done extracting features for %d / %d images' % (i, num_images)
    return imgs_features


def rgb2gray(rgb):
    """
    turn into gray image.
    :param rgb: RGB image
    :return: grayscale image
    """

