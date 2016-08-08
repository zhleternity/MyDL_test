import  cPickle as pickle
import numpy as np
import os


def load_CIFAR_batch(file_name):
    """载入数据集的一个batch"""
    with open(file_name,'r') as f:
        data_dict = pickle.load(f)
        x = data_dict['data']
        y = data_dict['labels']
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        y = np.array(y)
        return x, y

def load_CTFAR10(ROOT):
     """载入全部CIFAR数据集""" 