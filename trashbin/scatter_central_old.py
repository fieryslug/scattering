import numpy as np
import scipy as sp
from scipy import special
from matplotlib import pyplot as plt
import functools

#const
hbar = 1
k0 = 1
m = 1

#math things
Pi = np.pi
e = np.e
jl = special.spherical_jn
nl = special.spherical_yn

#legendre quad. const
QUAD = 'leg' #indicator
DEG = 40
CC = 1
X1, W1 = np.polynomial.legendre.leggauss(DEG)
K = (1+X1)/(1-X1) * CC
W = 2*CC*W1 / (1-X1)**2
KK = np.concatenate(([k0], K))

#mixed quadrature params
DEG2 = 20
R0 = 1

tmp_fl = [] #for testing
NUL = 104548791234

#prints current params
def printparams():
    print('=========PARAMS=========')
    print('-physical constants/params:')
    print('hbar = {} \nk0   = {} \nm    = {} \n'.format(hbar, k0, m))
    print('-quadrature params:')
    print('QUAD = {} \nDEG  = {} \nC    = {}'.format(QUAD, DEG, CC))

def summary(dlm = ' '):
    return ('hbar={}{}k0={}{}m={}'.format(hbar, dlm, k0, dlm, m) + '; ' + 'QUAD={}{}DEG={}{}C={}{}DEG2={}{}R0={}'.format(QUAD, dlm, DEG, dlm, CC, dlm, DEG2, dlm, R0))

#sets scattering parameters
def setparams(hbar1=hbar, k01=k0, m1=m):
    global hbar, k0, m, KK

    hbar = hbar1
    k0 = k01
    m = m1
    KK = np.concatenate(([k0], K))
    applychange()

def setquadparams(deg=DEG, c=CC, deg2=DEG2, r0=R0):
    global QUAD, DEG, CC, X1, W1, K, W, KK
    DEG = deg
    CC = c
    DEG2 = deg2
    R0 = r0
    setquadrature(quad=QUAD)
    if QUAD == 'leg':
        X1, W1 = np.polynomial.legendre.leggauss(DEG)
        K = (1+X1)/(1-X1) * CC
        W = 2*CC*W1 / (1-X1)**2
        KK = np.concatenate(([k0], K))
    if QUAD == 'lag':
        QUAD = 'lag'
        X1, W1 = np.polynomial.laguerre.laggauss(DEG)
        K = CC * X1
        W = CC * np.exp(X1) * W1
        KK = np.concatenate(([k0], K))
    applychange()

def setquadrature(quad=QUAD):
    global QUAD, CC, X1, W1, K, W, KK
    assert quad in ['leg', 'lag', 'mixed']
    if quad == 'leg':
        QUAD = 'leg'
        X1, W1 = np.polynomial.legendre.leggauss(DEG)
        K = (1+X1)/(1-X1) * CC
        W = 2*CC*W1 / (1-X1)**2
        KK = np.concatenate(([k0], K))
    if quad == 'lag':
        QUAD = 'lag'
        X1, W1 = np.polynomial.laguerre.laggauss(DEG)
        K = CC * X1
        W = CC * np.exp(X1) * W1
        KK = np.concatenate(([k0], K))
    if quad == 'mixed':
        QUAD = 'mixed'
        X1, W1 = np.polynomial.legendre.leggauss(DEG)
        Kp = R0/2 + R0/2 * X1
        Wp = R0/2 * W1

        X2, W2 = np.polynomial.laguerre.laggauss(DEG2)
        Kl = CC * X2 + R0
        Wl = CC * np.exp(CC * X2) * W2
        
        K = np.concatenate((Kp, Kl))
        W = np.concatenate((Wp, Wl))
        KK = np.concatenate(([k0], Kp))
    applychange()

def applychange():
    global setparams, setquadparams, setquadrature
    setparams = functools.partial(setparams, hbar1=hbar, k01=k0, m1=m)
    setquadparams = functools.partial(setquadparams, deg=DEG, c=CC, deg2=DEG2, r0=R0)
    setquadrature = functools.partial(setquadrature, quad=QUAD)
    
#finite legendre quadrature 
def leg_quad(f, a, b, deg=DEG):
    X, W = np.polynomial.legendre.leggauss(deg)
    s = 0
    A = (b+a)/2
    B = (b-a)/2
    
    for i in range(deg):
        s += W[i] * f(A + B*X[i])
    return B * s

#quadrature for (0, inf)
def leg_quad_halfinf(f):
    #print('DEG={}, C={}, QUAD={}'.format(DEG, CC, QUAD))
    s = 0
    for i in range(DEG):
        s += W[i] * f(K[i])
    return s

#V_l[j][i], O(n^3)
def V_l(V, l):
    res = []
    jl = special.spherical_jn
    n = len(KK)
    print('in Vl: {}'.format(n))
    for j in range(n):
        tmp = []
        for i in range(n):
            #print('({}, {})'.format(KK[i], KK[j]))
            Vji = leg_quad_halfinf(lambda r: r**2 * jl(l, KK[i]*r) * V(r) * jl(l, KK[j]*r))
            tmp.append( Vji)
        res.append(tmp)
    return np.array(res)

def V_l_box(V, l, r0):
    print('invoked; r0={}'.format(r0))
    res = []
    n = len(KK)
    for j in range(n):
        tmp = []
        for i in range(n):
            #print('({}, {})'.format(KK[i], KK[j]))
            Vji = leg_quad(lambda r: r**2 * jl(l, KK[i]*r) * V(r) * jl(l, KK[j]*r), 0, r0)
            tmp.append(Vji)
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
def T_conj(V, maxl, isbox=False, r0=0):
    assert not ((isbox and r0==0) or ((not isbox) and r0 != 0))
    global tmp_fl
    R = []
    tmp_fl = []
    for l in range(0, maxl+1):
        print('l={}'.format(l))
        Vl = V_l(V, l)
        if isbox:
            Vl = V_l_box(V, l, r0)
        Ul = Vl[:, 0]
        Dl = D_l(Vl)
        print('Ul {}'.format(Ul.shape))
        print('Dl {}'.format(Dl.shape))
        print('Vl {}'.format(Vl.shape))
        Tl = np.matmul(np.linalg.inv(I(len(Ul))-Dl), Ul)  #inv O(n^3)
        R.append(Tl)
        tmp_fl.append(-(2*m*k0)/(hbar**2)*Tl[0])
    return np.array(R)

#scattering amplitude approx. up to l=maxl, O(l*n^3)
def amplitude_f(V, maxl, isbox=False, r0=0):
    Tconj = T_conj(V, maxl, isbox, r0)
    return lambda theta: (-4*Pi**2*m/(hbar**2) * 1/(2*Pi**2) * sum([(2*l+1)*Tconj[l][0] * special.legendre(l)(np.cos(theta)) for l in range(maxl+1)]))

#update; testing
def amplitude_f_2(V, maxl, isbox=False, r0=0):
    Tconj = T_conj(V, maxl, isbox, r0)
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
