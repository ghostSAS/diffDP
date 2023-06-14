from autograd import grad
from autograd import make_jvp

def get_AB(fun, atX, atU):
    # A = make_jvp(fun)(atX)
    # B = make_jvp(fun)(atU)
    A = grad(fun)
    B = grad(fun)
    
    return A, B
    