import numpy as np
from scipy import special

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

def boxpot(r0, V0):
    return lambda r: V0 * (1 - np.heaviside(r-r0, 0))


class box:
    def __init__(self, r0, V0):
        self.r0 = r0
        self.V0 = V0
        self.V = boxpot(r0, V0)
        self.f = []

    @property
    def k_(self):
        return np.sqrt(k0**2 - (2*m*self.V0) / (hbar**2))

    def partial_f(self, l):
        k_ = self.k_
        r0 = self.r0
        L = k_*jl(l, k_*r0, True) / jl(l, k_*r0)
        #tandeltal = (jl(l, k_*r0)*k0*jl(l, k0*r0, True) - k_*jl(l, k_*r0, True)*jl(l, k0*r0)) / (jl(l, k_*r0)*k0*nl(l, k0*r0, True) - k_*jl(l, k_*r0, True)*nl(l, k0*r0))
        tandeltal = (k0*jl(l, k0*r0, True) - L*jl(l, k0*r0)) / (k0*nl(l, k0*r0, True) - L*nl(l, k0*r0))
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


class general:
    def __init__(self, hbar, k0, m, V, r0):
        self.k0 = k0
        self.hbar = 1
        self.m = m
        self.V = V
        self.r0 = r0
        self.U = lambda r: (2*self.m*self.V(r) / self.hbar**2)
        self.f = []
    #logarithmic derivative
    def L(self, l, dr=0.001):
        r = 0
        u = 0
        u_ = 0.01
        u__ = 0
        if l == 0:
            u_ = 1
        while r <= self.r0:
            if abs(r) <= 1e-4:
                u__ = 0
            else:
                u__ = (self.U(r) + l*(l+1)/r**2 - self.k0**2) * u
            u_ += u__ * dr
            u += u_ * dr
            r += dr
            print('u={}, u_={}'.format(u, u_))
        return u_ / u

    def partial_f(self, l):
        Lg = self.L(l)
        k = self.k0
        R = self.r0
        tandeltal = (jl(l, k*R) + k*R*jl(l, k*R, True) - R*jl(l, k*R)*Lg) / (nl(l, k*R) + k*R*nl(l, k*R, True) - R*nl(l, k*R)*Lg)
        deltal = np.arctan(tandeltal)
        return np.exp(1j*deltal) * np.sin(deltal)

    def compute_f(self, maxl):
        self.f = []
        for l in range(0, maxl+1):
            self.f.append(self.partial_f(l))

    def amplitude_f(self, maxl):
        self.compute_f(maxl)
        def f(theta):
            S = 0
            for l in range(maxl+1):
                S += special.legendre(l)(np.cos(theta)) * (2*l+1) * self.f[l] / self.k0
            return S
        return f
