import numpy as np
import scipy as sp
from scipy import special
from matplotlib import pyplot as plt


#const
hbar = 1
k0 = 1
m = 1
Pi = np.pi
e = np.e

#legendre quad. const
DEG = 40
CC = 10

X1, W1 = np.polynomial.legendre.leggauss(DEG)
K = (1+X1)/(1-X1) * CC
W = 2*CC*W1 / (1-X1)**2

KK = np.concatenate(([k0], K))


tmp_fl = [] #for testing

#sets scattering parameters
def setparams(hbar1=hbar, k01=k0, m1=m):
    global hbar, k0, m, KK
    hbar = hbar1
    k0 = k01
    m = m1
    KK = np.concatenate(([k0], K))

def setquadparams(deg=DEG, c=CC):
    global DEG, CC, X1, W1, K, W, KK
    DEG = deg
    CC = c
    X1, W1 = np.polynomial.legendre.leggauss(DEG)
    K = (1+X1)/(1-X1) * CC
    W = 2*CC*W1 / (1-X1)**2
    KK = np.concatenate(([k0], K))

#legendre quadrature 
def leg_quad(f, a, b, deg=DEG):
    X, W = np.polynomial.legendre.leggauss(deg)
    s = 0
    A = (b+a)/2
    B = (b-a)/2
    
    for i in range(deg):
        s += W[i] * f(A + B*X[i])
    return B * s

#legendre quadrature for (0, inf)
def leg_quad_halfinf(f, deg=DEG, C=CC):
    X1, W1 = np.polynomial.legendre.leggauss(deg)
    K = (1+X1)/(1-X1) * C
    W = 2 * C * W1 / (1-X1)**2
    s = 0
    for i in range(deg):
        s += W[i] * f(K[i])

    return s

#V_l[j][i], O(n^3)
def V_l(V, l, deg=DEG, C=CC):
    X1, W1 = np.polynomial.legendre.leggauss(deg)
    K = (1+X1)/(1-X1) * C
    W = 2 * C * W1 / (1-X1)**2
    KK = np.concatenate(([k0], K))
    
    res = []
    #jl = lambda u: special.spherical_jn(l, u)
    jl = special.spherical_jn
    n = len(KK)

    for j in range(n):
        tmp = []
        for i in range(n):
            Vji = leg_quad_halfinf(lambda r: r**2 * jl(l, KK[i]*r) * V(r) * jl(l, KK[j]*r))
            tmp.append( Vji)
        res.append(tmp)
    return np.array(res)

#D_l[j][i], O(n^3)
def D_l(Vl, deg=DEG, C=CC):
    res = []
    N = len(KK) #N=n+1
    for j in range(N):
        tmp = []
        for i in range(N):
            if i == 0:
                S = 0
                for a in range(1, N):
                    S += (W[a-1] * KK[0]**2 * Vl[j][0])/(KK[0]**2-KK[a]**2)  #prev  S += (KK[0]**2 * Vl[j][0])/(W[a-1] * KK[0]**2-KK[a]**2)
                tmp.append(-4*m/(Pi*hbar**2) * S - 1j*(2*m*KK[0]*Vl[j][0])/(hbar**2))
            else:
                tmp.append(4*m/(Pi*hbar**2) * (W[i-1] * KK[i]**2 * Vl[j][i])/(KK[0]**2-KK[i]**2))
        res.append(tmp)
    return np.array(res)

#T_l's up to l=maxl, O(l*n^3)
def T_conj(V, maxl):
    global tmp_fl
    R = []
    tmp_fl = []
    for l in range(0, maxl+1):
        print('l={}'.format(l))
        Vl = V_l(V, l)
        Ul = Vl[:, 0]
        Dl = D_l(Vl)
        Tl = np.matmul(np.linalg.inv(I(len(Ul))-Dl), Ul)  #inv O(n^3)
        R.append(Tl)
        tmp_fl.append(-(2*m*k0)/(hbar**2)*Tl[0])
    return np.array(R)

#scattering amplitude approx. up to l=maxl, O(l*n^3)
def amplitude_f(V, maxl):
    Tconj = T_conj(V, maxl)
    return lambda theta: (-4*Pi**2*m/(hbar**2) * 1/(2*Pi**2) * sum([(2*l+1)*Tconj[l][0] * special.legendre(l)(np.cos(theta)) for l in range(maxl+1)]))

#update; testing
def amplitude_f_2(V, maxl):
    Tconj = T_conj(V, maxl)
    def f(theta):
        S = 0
        for l in range(maxl+1):
            S += (2*l+1) * Tconj[l][0] * special.legendre(l)(np.cos(theta))
        return -4*Pi**2 *m/(hbar**2) * 1/(2*Pi**2) * S
    return f   
#identity matrix
def I(n):
    res = []
    for j in range(n):
        tmp = []
        for i in range(n):
            if i==j:
                tmp.append(1)
            else:
                tmp.append(0)
        res.append(tmp)
    return np.array(res)

#modulus squared
def modsq(z):
    return np.abs(z)**2

#total cross section
def stot(f):
    return 2*Pi*leg_quad(lambda theta: modsq(f(theta))*np.sin(theta), 0, np.pi)
