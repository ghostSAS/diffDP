import time as time
import matplotlib.pyplot as plt
import numpy as np

import pickle

import sys
sys.path.append('.')

from CDDP import *

dynamics_sym = simple_pen_sym
dynamics_fd = njit(simple_pen, cache=True)

n_x, n_u = 2, 1
x,u,dt = GetSyms(n_x, n_u)

N = 100

# f = sp.Matrix(RK4_sym(dt,x,u,dynamics_sym))
# f_x = f.jacobian(x)
# f_u = f.jacobian(u)

# f_sym = sympy_to_numba(f,[dt,x,u])
# f_x_sym = sympy_to_numba(f_x,[dt,x,u])
# f_u_sym = sympy_to_numba(f_u,[dt,x,u])

# with open('cache/f_x_sym.pkl', 'wb') as file:
#     pickle.dump(f_x_sym, file)
# with open('cache/f_u_sym.pkl', 'wb') as file:
#     pickle.dump(f_u_sym, file)
    
with open('cache/f_x_sym.pkl', 'rb') as file:
    f_x_sym = pickle.load(file)
with open('cache/f_u_sym.pkl', 'rb') as file:
    f_u_sym = pickle.load(file)


dt0, x0, u0 = np.array([.05]), np.random.rand(n_x), np.random.rand(n_u)
print(f_x_sym(dt0, x0, u0))
print(f_u_sym(dt0, x0, u0))

f_xu = finiteDiff(RK4,dynamics_fd, x0, u0, dt0, 1e-4)
print(f_xu)