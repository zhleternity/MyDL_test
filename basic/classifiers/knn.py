import numpy as np

class KNearestNeighbour:
    """L2 distance"""


    def __init__(self):
        pass
    def train(self, x, y):
        """
        train the classifier
        inputs:
        x:num_train x dimension array whiere each row is a training point
        y:a vector of length num_train ,where y[i] is the label for x[i, :]
        """

        self.x_train = x
        self.y_train = y


    def predict(self, x, k=1, num_loops=0):
