from numba import njit, jit
import numpy as np
import sympy as sp

import time

# import sys
# sys.path.append('../')

# from dynamics import *

def GetSyms(n_x, n_u):
  '''
      Returns matrices with symbolic variables for states and actions
      n_x: state size
      n_u: action size
  '''

  x = sp.IndexedBase('x')
  u = sp.IndexedBase('u')
#   dt = sp.symbols('dt')
  dt = sp.IndexedBase('dt')
  
#   xs = sp.Array([x[i] for i in range(n_x)])
#   us = sp.Array([u[i] for i in range(n_u)])
#   dts = sp.Array([dt[i] for i in range(1)])

  xs = sp.Matrix([x[i] for i in range(n_x)])
  us = sp.Matrix([u[i] for i in range(n_u)])
  
  return xs, us
    

def Constrain(cs, eps = 1e-4):
    '''
    Constraint via logarithmic barrier function
    Limitation: Doesn't work with infeasible initial guess.
    cs: list of constraints of form g(x, u) >= 0
    eps : parameters
    '''
    cost = 0
    for i in range(len(cs)):
        cost -= sp.log(cs[i] + eps)
    return 0.1*cost


def Bounded(vars, high, low, *params):
    '''
    Logarithmic barrier function to constrain variables.
    Limitation: Doesn't work with infeasible initial guess.
    '''
    cs = []
    for i in range(len(vars)):
        diff = (high[i] - low[i])/2
        cs.append((high[i] - vars[i])/diff)
        cs.append((vars[i] - low[i])/diff)
    return Constrain(cs, *params)


def SoftConstrain(cs, alpha = 0.01, beta = 10):
    '''
    Constraint via exponential barrier function
    cs: list of constraints of form g(x, u) >= 0
    alpha, beta : parameters
    '''
    cost = 0
    for i in range(len(cs)):
        cost += alpha*sp.exp(-beta*cs[i])
    return cost


def Smooth_abs(x, alpha = 0.25):
    '''
    smooth absolute value
    '''
    return sp.sqrt(x**2 + alpha**2) - alpha


@njit
def RK4(dt,x0,u,dynamics):
    """ Runge-Kutta method in the order of 4
    """
    # try:
    #     u = float(u)
    # except:
    #     pass
    # k1 = dynamics(0,x0,u,*args)
    # k2 = dynamics(dt/2,x0+dt*k1/2,u,*args)
    # k3 = dynamics(dt/2,x0+dt*k2/2,u,*args)
    # k4 = dynamics(dt,x0+dt*k3,u,*args)
    # print(u)
    dt = dt[0]
    k1 = dynamics(0,x0,u)
    k2 = dynamics(dt/2,x0+dt*k1/2,u)
    k3 = dynamics(dt/2,x0+dt*k2/2,u)
    k4 = dynamics(dt,x0+dt*k3,u)
    
    return x0+dt/6*(k1+2*k2+2*k3+k4)


# @njit
def RK4_sym(dt,x0,u,dynamics):
    """ Runge-Kutta method in the order of 4
    """
    
    dt = dt[0]
    k1 = dynamics(0,x0,u)
    k2 = dynamics(dt/2,x0+k1*dt/2,u)
    k3 = dynamics(dt/2,x0+k2*dt/2,u)
    k4 = dynamics(dt,x0+k3*dt,u)
    
    return x0+(k1+2*k2+2*k3+k4)*dt/6

def RK45_adptive(f,x0,u,T,epsilon = 0.0001,*args):
    """ currently give wrong numberical results
    """
    h = T/10
    t = 0
    w = x0
    i = 0
    # fprintf(’Step %d: t = %6.4f, w = %18.15f\n’, i, t, w)
    while t<T:
        h = min(h, 2-t)
        k1 = h*f(t,w,u,*args)
        k2 = h*f(t+h/4, w+k1/4,u,*args)
        k3 = h*f(t+3*h/8, w+3*k1/32+9*k2/32,u,*args)
        k4 = h*f(t+12*h/13, w+1932*k1/2197-7200*k2/2197+7296*k3/2197,u,*args)
        k5 = h*f(t+h, w+439*k1/216-8*k2+3680*k3/513-845*k4/4104,u,*args)
        k6 = h*f(t+h/2, w-8*k1/27+2*k2-3544*k3/2565+1859*k4/4104-11*k5/40,u,*args)
        w1 = w + 25*k1/216+1408*k3/2565+2197*k4/4104-k5/5
        w2 = w + 16*k1/135+6656*k3/12825+28561*k4/56430-9*k5/50+2*k6/55
        R = np.linalg.norm(w1-w2)/h
        delta = 0.84*(epsilon/R)**(1/4)
        if R<=epsilon:
            t = t+h
            w = w1
            i = i+1
            # fprintf(’Step %d: t = %6.4f, w = %18.15f\n’, i, t, w)
            h = delta*h
        else:
            h = delta*h
 
    return w

@njit
def get_traj( x0, U, dt):
    """ compute trajectory
    """
    x = x0
    traj = [x]
    N = U.shape[0]
    for n in range(N):
        x = RK4(x,U[n,:],dt)
        traj.append(x)
        
    return np.array(traj)

@jit(nopython=True)
def finiteDiff(fun, dynamics, x, u, dt, eps, vec='x'):
    '''
        Finite difference to get Jacobian
    '''
    n = fun(dt,x,u, dynamics).shape[0]

    if vec == 'x':
        m = x.shape[0]
        Jac = np.zeros((n,m))
        e = np.zeros(m)
        for i in range(m):
            e[i] = eps
            Jac[:,i] = (fun(dt,x+e,u,dynamics)-fun(dt,x-e,u,dynamics))/eps/2
            e[i] = 0
    elif vec == 'u':
        m = u.shape[0]
        Jac = np.zeros((n,m))
        e = np.zeros(m)
        for i in range(m):
            e[i] = eps
            Jac[:,i] = (fun(dt,x,u+e,dynamics)-fun(dt,x,u-e,dynamics))/eps/2
            e[i] = 0
    elif vec == 'dt':
        m = 1
        Jac = np.zeros((n,m))
        e = np.zeros(m)
        for i in range(m):
            e[i] = eps
            Jac[:,i] = (fun(dt+e,x,u,dynamics)-fun(dt-e,x,u,dynamics))/eps/2
            e[i] = 0
    
    return Jac

def sympy_to_numba(f, args, redu = True):
    '''
       Converts sympy matrix or expression to numba jitted function
    '''
    modules = [{'atan2':np.arctan2}, 'numpy']
    # modules = None

    if isinstance(f, sp.Matrix):
        #To convert all elements to floats
        m, n = f.shape
        f += 1e-64*np.ones((m, n))

        #To eleminate extra dimension
        if (n == 1 or m == 1) and redu:
            if n == 1: f = f.T
            f = sp.Array(f)[0, :]
            # f = njit(sp.lambdify(args, f, modules = modules), cache=True)
            f = njit(sp.lambdify(args, f, modules = modules))
            # f_new = lambda args: np.array(f(args))
            # return njit(f_new)
            return f
        
    f = sp.lambdify(args, f, modules = modules)
    return njit(f)


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.6f} seconds to execute.")
        return result
    return wrapper