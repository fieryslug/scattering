import numpy as np
from scipy import special
import utils

m = 1
hbar = 1
k0 = 1

jl = special.spherical_jn
nl = special.spherical_yn

def setparams(hbar1=hbar, k01=k0, m1=m):
    global hbar, k0, m
    hbar = hbar1
    k0 = k01
    m = m1

class box:
    def __init__(self, r0, V0):
        self.r0 = r0
        self.V0 = V0
        self.V = utils.box(r0, V0)
        self.f = []
    
    @property
    def k_(self):
        return np.sqrt(k0**2 - (2*m*self.V0) / (hbar**2))

    def partial_f(self, l):
        k_ = self.k_
        r0 = self.r0

        tandeltal = (jl(l, k_*r0)*k0*jl(l, k0*r0, True) - k_*jl(l, k_*r0, True)*jl(l, k0*r0)) / (jl(l, k_*r0)*k0*nl(l, k0*r0, True) - k_*jl(l, k_*r0, True)*nl(l, k0*r0))

        deltal = np.arctan(tandeltal)
        print('delta: {}, {}'.format(tandeltal, deltal))
        return np.exp(1j*deltal) * np.sin(deltal)

    def compute_f(self, maxl):
        self.f = []
        for l in range(maxl+1):
            self.f.append(self.partial_f(l))


    def amplitude_f(self, maxl):
        self.compute_f(maxl)
        def f(theta):
            S = 0
            for l in range(maxl+1):
                S += special.legendre(l)(np.cos(theta)) * (2*l+1) * self.f[l] / k0
            return S
        return f
        #return lambda theta: (1/k0) * sum([special.legendre(l)(np.cos(theta)) * (2*l+1) * self.f[l] for l in range(maxl+1)])    



