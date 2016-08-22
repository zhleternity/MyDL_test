#coding=utf-8
#__author__='eternity'


import numpy as np
import matplotlib.pyplot as plt
from nn.classifiers.neural_net import two_layer_net
from nn.gradient_check import evaluate_numerical_grad
from nn.classifier_trainer import ClassifierTrainer

#  This is a multi-layer NN on the dataset of CIFAR-10
#  we can ignore the initial set
plt.rcParams['figure.figsize'] = (10.0, 8.0)  #  set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#for auto-reloading external modules
#see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython


def relative_error(x, y):
    """
    return the relative error.
    """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
#  randomly initialize the model(acturally is the wights) and dataset

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

#  build the initial model
def init_toy_model():
    model = {}
    model['W1'] = np.linspace(-0.2, 0.6, num=input_size * hidden_size).reshape(input_size, hidden_size)
    model['b1'] = np.linspace(-0.3, 0.7, num=hidden_size)
    model['W2'] = np.linspace(-0.4, 0.1, num=hidden_size*num_inputs).reshape(hidden_size, num_classes)
    model['b2'] = np.linspace(-0.5, 0.9, num=num_classes)
    return model

#  contruct the used input data
def init_toy_data():
    x = np.linspace(-0.2, 0.5, num=num_inputs*input_size).reshape(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return x, y

model = init_toy_model()
x, y = init_toy_data()


#  forward-compute: obtain the score, loss, and the grads on the params
#  same as the svm and softmax

scores = two_layer_net(x, model, verbose=True)
print scores

correct_scores = [[-0.5328368, 0.20031504, 0.93346689],
 [-0.59412164, 0.15498488, 0.9040914],
 [-0.67658362, 0.08978957, 0.85616275],
 [-0.77092643, 0.01339997, 0.79772637],
 [-0.89110401, -0.08754544, 0.71601312]]

#  the diff between the computed score and the real score should be little
print 'the diff between the computed score and the real score is :' % np.sum(np.abs(scores - correct_scores))

#  forward compute: compute the loss(include the data loss and regularization)
regularization = 0.1
loss, _ = two_layer_net(x, model, y, regularization)
correct_loss = 1.38191946092
#  the diff also should be little
print 'the diff between the computed loss and the correct loss is : ' % np.sum(np.abs(loss - correct_loss))

#  back-propagation
#  To loss,we need compute the grads on W1,b1,W2,b2,it also needs do the grad-checkout

#  use the numerical grad to checkout
loss, grads = two_layer_net(x, model, y, regularization)
#  it is save that each param should be less than 1e-8
for param_name in grads:
    param_grad_num = evaluate_numerical_grad(lambda W: two_layer_net(x, model, y, regularization)[0],
                                             model[param_name], verbose=False)
    print  '%s maximum relative error: %e' % (relative_error(param_grad_num, grads[param_name]))




#  train the NN
#  we use fixed-step SGD and SGD with Momentum to minimum loss function
#  fixed-step SGD
model = init_toy_data()
trainer = ClassifierTrainer()
#  Caution:here,the data is man-made,and small scale,so set 'sample_batched' to False;
best_model, loss_history, _, _ = trainer.train(x, y, x, y,
                                               model, two_layer_net,regularization=0.001,
                                               learning_rate=1e-1, momentum=0.0,
                                               learning_rate_decay=1, update='sgd',
                                               sample_batches=False, num_epoches=100,
                                               verbose=False)
print 'Final loss with vanilla SGD: %f' % (loss_history[-1], )

#  SGD with momentum,you will see that the loss is less than above
model1 = init_toy_model()
trainer1 = ClassifierTrainer()
#  call the trainer to optimize the loss
#  Notice that we are using sample_batches=False,so we are performing SGD(no sampled batches of data)
best_model1, loss_history1, _, _ = trainer1.train(x, y, x, y,
                                                  model1, two_layer_net,
                                                  regularization=0.001, learning_rate=1e-1,
                                                  momentum=0.9, learning_rate_decay=1,
                                                  update='momentum', sample_batches=False,
                                                  num_epoches=100, verbose=False)
correct_loss = 0.494394
print 'Final loss with momentum SGD: %f. We get : %f' % (loss_history1[-1],correct_loss)



























































































