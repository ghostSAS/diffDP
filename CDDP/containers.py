import sympy as sp
import numpy as np
from numba import njit
from .utils import *
from .dynamics import *


class Dynamics:
    
    def __init__(self, num, dt):
        '''
        num: 
            1: simple pendulum
            2: cart pendulum
            3: furuta pendulum
            4: unicycle
        Dynamics container.
            f: Function approximating the dynamics.
            f_x: Partial derivative of 'f' with respect to state
            f_u: Partial derivative of 'f' with respect to action
            f_xx: second partial derivative of 'f' with respect to state
            f_uu: second partial derivative of 'f' with respect to action
            f_prime: returns f_x and f_u at once
        '''
        if num ==1:
            self.f = simple_pen
            self.n_x = 2
            self.n_u = 1
        elif num == 2:
            self.f = cart_pen
            self.n_x = 4
            self.n_u = 1
        elif num == 3:
            self.f = furuta_pen
            self.n_x = 4
            self.n_u = 1
        elif num == 4:
            self.f = unicycle
            self.n_x = 3
            self.n_u = 1
            
        self.f = njit(self.f, cache=True)
        self.dt = dt
        self.differential()
        
    # @timer_decorator
    def differential(self, x_eps = 1e-4, u_eps = 1e-4, dt_eps = 1e-5):
        '''
           Construct from a continuous time dynamics function
        '''

        # f = njit(self.f, cache=True)
        f = self.f
        
        # self.f_x = njit(lambda x,u,dt: finiteDiff(RK4,f, x, u, dt, x_eps, vec='x'))
        # self.f_u = njit(lambda x,u,dt: finiteDiff(RK4,f, x, u, dt, u_eps, vec='u'))
        # self.f_x(np.zeros(self.n_x),np.zeros(self.n_u),self.dt)
        # self.f_u(np.zeros(self.n_x),np.zeros(self.n_u),self.dt)
        
        f_x = njit(lambda x,u,dt: finiteDiff(RK4,f, x, u, dt, x_eps, vec='x'))
        f_u = njit(lambda x,u,dt: finiteDiff(RK4,f, x, u, dt, u_eps, vec='u'))
        f_dt = njit(lambda x,u,dt: finiteDiff(RK4,f, x, u, dt, dt_eps, vec='dt'))
        f_prime = njit(lambda x,u,dt:[f_x(x,u,dt), f_u(x,u,dt)])
        # f_x(np.zeros(self.n_x),np.zeros(self.n_u),np.zeros(1)+.1)
        # f_u(np.zeros(self.n_x),np.zeros(self.n_u),np.zeros(1)+.1)
        # f_prime(np.zeros(self.n_x),np.zeros(self.n_u),np.zeros(1)+.1)
        
        self.f_x = f_x
        self.f_u = f_u
        self.f_dt = f_dt
        self.f_prime = f_prime
        
        
    @staticmethod
    def SymContinuous(f, x, u, dt = 0.1):
        '''
           Construct from Symbolic continuous time dynamics
        '''
        return Dynamics.SymDiscrete(x + f*dt, x, u)
    
    
class Cost:

    def __init__(self, L, L_x, L_u, L_xx, L_ux, L_uu, Lf, Lf_x, Lf_xx):
        '''
           Container for Cost.
              L:  Running cost
              Lf: Terminal cost
        '''
        #Running cost and it's partial derivatives
        self.L = L
        self.L_x  = L_x
        self.L_u  = L_u
        self.L_xx = L_xx
        self.L_ux = L_ux
        self.L_uu = L_uu
        self.L_prime = njit(lambda x, u: (L_x(x, u), L_u(x, u), L_xx(x, u), L_ux(x, u), L_uu(x, u)))

        #Terminal cost and it's partial derivatives
        self.Lf = Lf
        self.Lf_x = Lf_x
        self.Lf_xx = Lf_xx
        self.Lf_prime = njit(lambda x: (Lf_x(x), Lf_xx(x)))


    @staticmethod
    def Symbolic(L, Lf, x, u):
        '''
           Construct Cost from Symbolic functions
        '''
        #convert costs to sympy matrices
        L_M  = sp.Matrix([L])
        Lf_M = sp.Matrix([Lf])

        #Partial derivatives of running cost
        L_x  = L_M.jacobian(x)
        L_u  = L_M.jacobian(u)
        L_xx = L_x.jacobian(x)
        L_ux = L_u.jacobian(x)
        L_uu = L_u.jacobian(u)

        #Partial derivatives of terminal cost
        Lf_x  = Lf_M.jacobian(x)
        Lf_xx = Lf_x.jacobian(x)

        #Convert all sympy objects to numba JIT functions
        funs = [L, L_x, L_u, L_xx, L_ux, L_uu, Lf, Lf_x, Lf_xx]
        for i in range(9):
          args = [x, u] if i < 6 else [x]
          redu = 0 if i in [3, 4, 5, 8] else 1
          funs[i] = sympy_to_numba(funs[i], args, redu)

        return Cost(*funs)

    @staticmethod
    def QR(Q, R, QT, x_goal, add_on = 0):
        '''
           Construct Quadratic cost
        '''
        x, u = GetSyms(Q.shape[0], R.shape[0])
        er = x - sp.Matrix(x_goal)
        L  = er.T@Q@er + u.T@R@u
        Lf = er.T@QT@er
        return Cost.Symbolic(L[0] + add_on, Lf[0], x, u)