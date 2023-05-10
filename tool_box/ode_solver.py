import autograd.numpy as np
# from numba import jit

# @jit(nopython=True)
def RK4(fun,x0,u,dt,*args):
    """ Runge-Kutta method in the order of 4
    """
    try:
        u = float(u)
    except:
        pass
    k1 = fun(0,x0,u,*args)
    k2 = fun(dt/2,x0+dt*k1/2,u,*args)
    k3 = fun(dt/2,x0+dt*k2/2,u,*args)
    k4 = fun(dt,x0+dt*k3,u,*args)
    
    return x0+dt/6*(k1+2*k2+2*k3+k4)

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

# @jit(nopython=True)
def get_traj(fun, x0, U, dt):
    """ compute trajectory
    """
    x = x0
    traj = [x]
    N = U.shape[0]
    for n in range(N):
        x = RK4(fun,x,U[n,:],dt)
        traj.append(x)
        
    return np.array(traj)
    
    
    