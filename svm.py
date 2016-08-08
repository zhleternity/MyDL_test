#!/usr/bin/python
#coding=utf-8
#__author__ = 'eternity'


import random
import numpy as np
from basic.datasets.data_util import load_CTFAR10
import matplotlib.pyplot as plt


#初始化
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

"""CIFAR-10数据集的载入与预处理"""
#加载原始的CIFAR-10图片数据集
cifar10_dir = 'basic/datasets/cifar-10-batches-py'
X_train, Y_train, X_test, Y_test = load_CTFAR10(cifar10_dir)

#see 训练集和测试集维度
print 'Training data shape: ', X_train.shape
print 'Training labels shape: ', Y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', Y_test.shape