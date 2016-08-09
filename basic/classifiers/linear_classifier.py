import numpy as np
from basic.classifiers.linear_svm import *
from basic.classifiers.softmax import *


class LinearClassifier:
    def __init__(self):
        self.W = None

    def train(self, x, y, learning_rate=1e-3, regularization=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        use SGD to train classifiers
        inputs:
        :param x: D x N array of training data ,each training point is a D-dimensional column.
        :param y: 1-dimensional array of length N with labels 0...k-1,for k classes.
        :param learning_rate: learning rate of optimization,float
        :param regularization: regularization strength,float.
        :param num_iters: number of steps to take when optimization,integer
        :param batch_size: number od training examples to use at each step,integer
        :param verbose: if true,print progress during optimization,boolean
        :outputs:
        a list containing the value of the loss function at each training iteration
        """
        dim, num_train = x.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            #lazily initialize W
            self.W = np.