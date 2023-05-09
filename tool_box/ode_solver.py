def RK45(fun,x,u,x0,dt,*args):
    k1 = fun(x,u,*args)
    k2 = fun(x0+dt*k1/2,u,*args)
    k3 = fun(x0+dt*k2/2,u,*args)
    k4 = fun(x0+dt*k3,u,*args)
    
    return x0+dt/6*(k1+2*k2+2*k3+k4)
    
    