#!/usr/bin/python
#coding=utf-8
#__author__ = 'eternity'

import cv2
import random
import numpy as np
from basic.datasets.data_util import load_CTFAR10
import matplotlib.pyplot as plt
from scipy import *


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

#可视化一下图片集,每个类展示一些图片
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog','horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxes = np.flatnonzero(Y_train == y)
#     idxes = np.random.choice(idxes, samples_per_class, replace=False)
#     for i, idx in enumerate(idxes):
#       plt_idx = i * num_classes + y + 1
#       plt.subplot(samples_per_class, num_classes, plt_idx)
#       plt.imshow(X_train[idx].astype('uint8'))
#       plt.axis('off')
#       if i == 0:
#           plt.title(cls)
# plt.show()

#extract the training set ,validation set ,and test set
num_training = 49000
num_validation = 1000
num_test = 1000

#fetch the image
mask = range(num_training, num_training + num_validation)
x_val = X_train[mask]
y_val = Y_train[mask]

mask = range(num_training)
x_train = X_train[mask]
y_train = Y_train[mask]

mask = range(num_test)
x_test = X_test[mask]
y_test = Y_test[mask]

print 'Train data shape: ', x_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', x_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape :', x_test.shape
print 'Test labels shape: ', y_test.shape

#preprocessing:change the data in view of column
x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_val = np.reshape(x_val, (x_val.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

print 'Training data shape: ', x_train.shape
print 'Validation data shape: ', x_val.shape
print 'Test data shape: ', x_test.shape

#preprocessing:substact the mean of image
mean_image = np.mean(x_train, axis=0)
print mean_image[:10]
plt.figure(figsize=(4, 4))
plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))



