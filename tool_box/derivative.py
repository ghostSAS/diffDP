from autograd import jacobian
from autograd import make_jvp


def get_AB(fun, atX, atU):
    A = make_jvp(fun)(atX)
    B = make_jvp(fun)(atU)
    
    return A, B
    