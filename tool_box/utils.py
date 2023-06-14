from numba import njit
import numpy as np
# import sys
# sys.path.append('../')

# from dynamics import *


# dynamics = njit(cart_pen, cache = True)
# dynamics = cart_penJIT
    
@njit
def RK4(x0,u,dt,dynamics):
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
    k1 = dynamics(0,x0,u)
    k2 = dynamics(dt/2,x0+dt*k1/2,u)
    k3 = dynamics(dt/2,x0+dt*k2/2,u)
    k4 = dynamics(dt,x0+dt*k3,u)
    
    return x0+dt/6*(k1+2*k2+2*k3+k4)

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

@njit
def finiteDiff(fun, x, u, dt, eps):
    '''
        Finite difference to get Jacobian
    '''
    
    fun0 = fun(x,u,dt)
    n = fun0.shape[0]
    X = np.append(x,u)
    m = X.shape[0] 
    xdim = x.shape[0]

    Jac = np.zeros((n,m))
    
    e = np.zeros(m)
    for i in range(m):
        e[i] = eps
        tmp1 = X+e
        tmp2 = X-e
        Jac[:,i] = (fun(tmp1[:xdim],tmp1[xdim:],dt)-fun(tmp2[:xdim],tmp2[xdim:],dt))/eps/2
        
        e[i] = 0
    
    return Jac