from ilqr import *
import numpy as np

xs,us = utils.GetSyms(4,2)

print(xs, us)


def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)

fun = sigmoid