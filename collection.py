import numpy as np




def yukawa(g, a):
    def V(r):
        return g * (np.exp(-a*r) / (r))
    return V

def box(V0, r0):
    def V(r):
        return V0 * (1-np.heaviside(r-r0, 0))
    return V

def bump(g, a):
    def V(r):
        return g * np.exp(-(a*r)**2)
    return V

def wavelike(g, a):
    def V(r):
        return g * np.sin(a*r)**2 / (a*r)**2
    return V

def wavelike2(g, a, b):
    def V(r):
        return g * np.cos(a*r) * np.exp(-(b*r)**2)
    return V
