# import path
import sys
sys.path.append('../')

import autograd.numpy as np
import time as time

from tool_box import *

import visulization.static as vsta
import visulization.animation as vani

dynamics = cart_pen


# print(dyn.RK4(np.zeros(4),1,.01))

finiteDiff(RK4,np.zeros(4),np.array([1]),.01,1e-4)

# x = np.random.rand(4)

# u = np.array([1])
# print(float(u))


# print(np.append(x,u))
