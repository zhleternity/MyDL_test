#!/usr/bin/python
#coding=utf-8
#__author__ = 'eternity'

import time
import random
import numpy as np
from basic.datasets.data_util import load_CTFAR10
from basic.classifiers.linear_svm import *
import matplotlib.pyplot as plt
from scipy import *
from  basic.check_gradient import *


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
# time.sleep(1.0)
x_train = x_train - mean_image
x_val = x_val - mean_image
x_test = x_test - mean_image

#add the column of '1'
x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))]).T
x_val = np.hstack([x_val, np.ones((x_val.shape[0], 1))]).T
x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))]).T
print x_train.shape, x_val.shape, x_test.shape

#svm
#evaluate the efficiency of svm_loss_naive

#produce the initial weights of svm
W = np.random.randn(10, 3073) * 0.0001
#compute the gradient and loss under the wieght W
loss, gradient = svm_loss_naive(W, x_train, y_train, 0.00001)
print 'loss: %f' % (loss, )


loss, gradient = svm_loss_naive(W, x_train, y_train, 0.0)
#gradient check :check out weather the numerical gradient and analytic gradient is identical,because the latter is fast,
# but eary to error
f = lambda w:svm_loss_naive(w, x_train, y_train,0.0)[0]
grad_numerical = gradient_check_sparse(f, W, gradient, 10)

#two methods to cpmpute the loss of svm:generately,vectorize method is faster
#naive loss of non-vectorize svm ,loss computing
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(W, x_train, y_train, 0.00001)
toc = time.time()
print 'non-vectorize:loss %e timeout %fs' % (loss_naive, toc - tic)
#vectorzie
tic = time.time()
loss_vectorize, _ = svm_loss_vectorized(W, x_train, y_train, 0.00001)
toc = time.time()
print 'vectorize:loss %e timeout %fs' % (loss_vectorize, toc - tic)
#if your implementation is right ,the two value is same
print 'difference of two methods: %f ' % (loss_naive - loss_vectorize)

#sgd


