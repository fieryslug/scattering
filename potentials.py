import numpy as np

def yukawa(g, a):
    def V(r):
        return g * (np.exp(-r/a) / (r))
    return V

def box(V0, r0):
    def V(r):
        return V0 * (1-np.heaviside(r-r0, 0))
    return V

def bump(g, a):
    def V(r):
        return g * np.exp(-(r/a)**2)
    return V

def wavelike(g, a, b):
    def V(r):
        if abs(r) <= 1e-4:
            return g * a**2 / b**2
        else:
            return g * np.sin(a*r)**2 / (b*r)**2
    return np.vectorize(V, otypes=[float])

def wavelike2(g, a, b):
    def V(r):
        return g * np.cos(r/b) * np.exp(-(r/a)**2)
    return V

def shell(g, a, c):
    b = 0.1
    def V(r):
        return g * np.tanh(r/b) * np.exp(-((r-c)/a)**2)

    return V
