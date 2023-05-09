import autograd.numpy as np

def simple_pen(x,u, m=1, l=.5, b=.1):
    g = 9.8
    # b: damping factor
    
    xdot = np.array([x[1], 
                     (u-m*g*l*np.sin(x[0])-b*x[1])/m/l/l])
    
    # return: ddt[angle, angular velocity]
    return xdot


def cart_pen(x,u,m=.2,M=.5,l=.3,b=.1,I=.006):
    # model from: https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
    g = 9.8
    
    [xi1, xi2, xi3, xi4] = x
    xi5 = u
    cos = np.cos()
    sin = np.sin()
    
    xdot  = np.array([xi3,
        1/(m+M*(1-cos(xi2)**2))*(l*M*sin(xi2)*xi4**2+xi5+M*g*cos(xi2)*sin(xi2)),
        xi4,
        -1/(l*m+l*M*(1-cos(xi2)**2))*(l*M*cos(xi2)*sin(xi2)*(xi4)**2+xi5*cos(xi2)+(m+M)*g*sin(xi2))])
    
    # return: ddt[cart_position, cart_velocity, arm_angle, arm_rate]
    return xdot


def furuta_pen(x,u,m2=.127,l1=.2,l2=.3):
    j1 = 0.0012
    lc2 = 0.15
    beta1 = 0.015
    beta2 = 0.002
    r = 2.6
    kc = 0.00768
    kv = kc
    kr = 70
    
    g = 9.8

    comm = r/(kc*kr)
    a1 = comm*(j1+m2*l1**2)
    a2 = comm*m2*l2**2/3
    a3 = comm*m2*l1*l2/2
    a4 = comm*m2*lc2*g
    a5 = beta1*comm + kv
    a6 = beta2*comm
    
    [xi1, xi2, xi3, xi4] = x
    xi5 = u
    cos = np.cos()
    sin = np.sin()
    
    U = np.array([[a1+a2*sin(xi2)**2, a3*cos(xi2)],
        [a3*cos(xi2), a2]])
    V = np.array([a5*xi3-a3*sin(xi2)*xi4**2+2*a2*sin(xi2)*cos(xi2)*xi3*xi4,
        -a2*sin(xi2)*cos(xi2)*xi3**2+a6*xi4-a4*sin(xi2)])

    xdot = np.linalg.inv(U)@(np.array([xi5, 0]) - V)
    
    # return: ddt[arm1_theta, arm2_theta, arm1_rate, arm2_rate]
    return xdot
    