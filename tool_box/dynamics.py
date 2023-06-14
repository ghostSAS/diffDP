# import autograd.numpy as np
import numpy as np
import sympy
from numba import jit, njit
# from .utils import *

@jit(nopython=True)
def simple_pen(t,x,u, m=1, l=.5, b=.1):
    """ simple pendulum with single arm rotating around a single point
        b: damping factor
        
        return: ddt[angle, angular velocity]
    """
    g = 9.81
    
    xdot = np.array([x[1], 
                     (u-m*g*l*np.sin(x[0])-b*x[1])/m/l/l])
    
    return xdot

def simple_pen_get_AB():
    x = sympy.symbols('x1,x2')
    u = sympy.symbols('u')
    
    sin = sympy.sin
    
    
def cart_pen(t,x,u,M=.5,m=.2,l=.3,b=.1,I=.006):
    """ cart pendulum 
        model from: https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling

        return: ddt[cart_position, arm_angle, cart_velocity, arm_rate]
    """
    # 
    g = 9.81
    [xi1, xi2, xi3, xi4] = x
    xi5 = u[0]
    m1 = M
    m2 = m
    xdot  = np.array([xi3,
        xi4,
        1/(m1+m2*(1-np.cos(xi2)**2))*(l*m2*np.sin(xi2)*xi4**2+xi5+m2*g*np.cos(xi2)*np.sin(xi2)),
        -1/(l*m1+l*m2*(1-np.cos(xi2)**2))*(l*m2*np.cos(xi2)*np.sin(xi2)*(xi4)**2+xi5*np.cos(xi2)+(m1+m2)*g*np.sin(xi2))])
    
    return xdot

@jit(nopython=True)
def furuta_pen(t,x,u,m2=.127,l1=.2,l2=.3):
    j1 = 0.0012
    lc2 = 0.15
    beta1 = 0.015
    beta2 = 0.002
    r = 2.6
    kc = 0.00768
    kv = kc
    kr = 70
    
    g = 9.81

    comm = r/(kc*kr)
    a1 = comm*(j1+m2*l1**2)
    a2 = comm*m2*l2**2/3
    a3 = comm*m2*l1*l2/2
    a4 = comm*m2*lc2*g
    a5 = beta1*comm + kv
    a6 = beta2*comm
    
    [xi1, xi2, xi3, xi4] = x
    xi5 = u
    cos = np.cos
    sin = np.sin
    
    U = np.array([[a1+a2*sin(xi2)**2, a3*cos(xi2)],
        [a3*cos(xi2), a2]])
    V = np.array([a5*xi3-a3*sin(xi2)*xi4**2+2*a2*sin(xi2)*cos(xi2)*xi3*xi4,
        -a2*sin(xi2)*cos(xi2)*xi3**2+a6*xi4-a4*sin(xi2)])

    xdot = np.linalg.inv(U)@(np.array([xi5, 0]) - V)
    
    # return: ddt[arm1_theta, arm2_theta, arm1_rate, arm2_rate]
    return xdot
    
    
