#!/usr/bin/env python
#coding=utf-8
#__author__ = 'eternity'


import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegressionCV

import matplotlib.pyplot as plt

#  using NN to realize the non-linear separation

#  produce an random distribution of planar point manually,and plot them.
np.random.seed(0)
x, y = make_moons(200, noise=0.20)
# plt.scatter(x[:, 0], x[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

#  define a function as the decision boundary
if __name__ == '__main__':
    if __name__ == '__main__':
        if __name__ == '__main__':
            def decision_boundary(pred_func):
                #  set the max and min value , and fill the boundary
                x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
                y_min, y_max = y[:, 1].min() - 0.5, y[:, 1].max() + 0.5
                h = 0.01
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

                #  predict using the predict func
                z = pred_func(np.c_[xx.ravel(), yy.ravel()])  #  ravel():Return a contiguous flattened array
                #  np.c_: concatenate alone the second axis

                z = z.reshape(xx.shape)

                #  use LR to classify
                #  Firstly,see the LR effect
                clf = LogisticRegressionCV()
                clf.fit(x, y)

                #  Obviously,the classification result is not satisfied,so we try to use NN to do it
                num_examples = len(x)  #  number of samples
                nn_input_dim = 2  #  dims of input
                nn_output_dim = 2  #  number of output classes

                #GD params
                lr = 0.01  #  learning rate
                reg = 0.01  #  regularization

                #  define loss func
                def calculate_loss(model):
                    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

                    # forward computing
                    z1 = x.dot(W1) + b1
                    a1 = np.tanh(z1)
                    z2 = a1.dot(W2) + b2
                    exp_scores = np.exp(z2)
                    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

                    # compute loss
                    












































