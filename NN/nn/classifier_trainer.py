#!/usr/bin/env python
#  _*_ coding: utf-8 _*_
#__author__ = 'eternity'


import numpy as np


class ClassifierTrainer(object):
    """The trainer class performs SGD with momentum on a cost fucntion"""

    def __init__(self):
        self.step_cache = {}  #  for storing velocities(speed) in momentum update
    def train(self, X, y, X_val, y_val, model, loss_func,
              regularization=0.0, learning_rate=1e-2, momentum=0,
              learning_rate_decay=0.95, update='momentum', sample_batches=True,
              num_epoches=30, batch_size=100, accuracy_frequency=None, verbose=False):
        """
        Optimize the params of a model to minimize a loss func .
        We use training data X and y to compute the loss and grads , and periodically check the accuracy on the validation data.
        Inputs:
        :param X: Array of training data;each X[i] is a training sample.
        :param y: Vector of training labels;y[i] gives the label for X[i].
        :param X_val: Array of validation data.
        :param y_val: Vector of validation labels.
        :param model: Dict that maps params names to param values.Each param value is a numpy array.
        :param loss_func: A func that can be called in the following ways:
           scores = loss_func(X, model, reg=regularization)
           loss, grads = loss_func(X, model, y, reg=regularization)
        :param regularization: Regularization strength,which will be passed to the loss func.
        :param learning_rate: Initial learning rate .
        :param momentum: Parameter to use for momentum updates.
        :param learning_rate_decay: The Learning rate will be multiplied by it after each epoch.
        :param update: The update rule to use.One of ''sgd ,'momentum','rmsprop'.
        :param sample_batches: If True,use a mini-batch of data for each parameter update(Stochastic Gradient Descent);
           If False, use the entire training set for each parameter update(Gradient Descent).
        :param num_epochs: The number of epochs to take over the training data.
        :param batch_size: The number of training samples to use at each iteration.
        :param accuracy_frequency: If set to an integer, we compute the training and validation set error after every
              accuracy_frequency iteration.
        :param verbose: If True ,print status after each epoch.
        Returns a tuple of :
        --best_model: the model that got the highest validation accuracy during training.
        --loss_history: List containing the value of the loss func at each iteration.
        --train_accuracy_history: List storing the training set accuracy at each epoch.
        --val_accuracy_history: List storing the validation set accuracy at each epoch.

        """
































