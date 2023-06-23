# import path
import sys
sys.path.append('.')

import autograd.numpy as np

from CDDP import *



dynS = Dynamics(num=2,dt=.01)

@timer_decorator
def evaluate_f_xu(dynS):
    # print(dynS.f_x(np.random.rand(4), np.random.rand(1), .02))
    # print(dynS.f_u(np.random.rand(4), np.random.rand(1), .02))
    # print(dynS.f_dt(np.random.rand(4), np.random.rand(1), np.zeros(1)+.1))
    print(dynS.f_dt(np.zeros(4)+1, np.zeros(1)+1, np.zeros(1)+.1))
    
f = evaluate_f_xu
f(dynS)

