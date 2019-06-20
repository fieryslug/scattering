import numpy as np
import scatter_central as sc
import partial_wave as an
from scipy import special
from matplotlib import pyplot as plt


k0 = sc.k0
hbar = sc.hbar
m = sc.m
W = sc.W
KK = sc.KK


def testeq():
    b = an.box(1, -1)
    V = b.V

    Tconj = sc.T_conj(V, 10)
    N = sc.DEG+1
    #L=0

    flag = True
    for L in range(0, 11):
        V0 = sc.V_l(V, L)
        for J in range(0, 41):
            print('J={}'.format(J))
            #print(Tconj[0][J])

            S = 0
            for i in range(1, N):
                S += W[i-1] * (KK[i]**2*V0[J][i]*Tconj[L][i] - KK[0]**2*V0[J][0]*Tconj[L][0]) / (KK[0]**2 - KK[i]**2)

            TTT = V0[J][0] + 4*m/(np.pi*hbar**2) * S - 1j*2*m*k0/(hbar**2) * V0[J][0]*Tconj[L][0]

            f =  (abs((TTT - Tconj[L][J])/TTT) < 1e-13) 
            flag = f
            print(f)

    print('overall {}'.format(flag))

def compare(hbar=1, k0=1, m=1, r0=1, V0=-1, deg=40, c=10, maxl=10):
    an.setparams(hbar, k0, m)
    sc.setparams(hbar, k0, m)
    sc.setquadparams(deg, c)
    print(len(sc.KK))
    b = an.box(r0, V0)
    V = b.V

    f = sc.amplitude_f(V, maxl, isbox=False, r0=0)
    fan = b.amplitude_f(maxl)

    theta = np.linspace(0, np.pi, 10000)
    plt.plot(theta, sc.modsq(f(theta)), theta, sc.modsq(fan(theta)))
    print('sc: {}'.format(sc.stot(f) - np.pi*4/k0 * np.imag(f(0))))
    print('an: {}'.format(sc.stot(fan) - np.pi*4/k0 * np.imag(fan(0))))
    print('----fl----')
    for l in range(maxl+1):
        print('l={}'.format(l))
        print('sc: {}'.format(sc.tmp_fl[l]))
        print('an: {}'.format(b.f[l]))
        print()
    plt.xlabel('hbar={}, k0={}, m={}, r0={}, V0={}, maxl={}, deg_legendre={}'.format(hbar, k0, m, r0, V0, maxl, deg))
    plt.show()

V_sample = an.boxpot(5, -1)
def sc_l_deg(hbar=1, k0=1, m=1, V=V_sample, degs=[30, 40, 50, 60, 70], c=10, maxl=10):
    an.setparams(hbar, k0, m)
    sc.setparams(hbar, k0, m)
    theta = np.linspace(0, np.pi, 10000)

    b = an.box(5, -1)

    for deg in degs:
        sc.setquadparams(deg, c)
        f = sc.amplitude_f(V, maxl, isbox=True, r0=5)
        plt.plot(theta, sc.modsq(f(theta)), label=str(deg))

    fan = b.amplitude_f(maxl)
    plt.plot(theta, sc.modsq(fan(theta)), label='partial wave')
    plt.legend()
    plt.xlabel('hbar={}, k0={}, m={}, V={}, maxl={}, degs={}'.format(hbar, k0, m, 'box(5, -1)', maxl, degs))
    plt.show()

def sc_l_maxl(hbar=1, k0=1, m=1, V=V_sample, deg=40, c=10, maxls=[5, 10, 20, 30, 40]):
    an.setparams(hbar, k0, m)
    sc.setparams(hbar, k0, m)
    theta = np.linspace(0, np.pi, 10000)

    b = an.box(5, -1)

    for maxl in maxls:
        #sc.setquadparams(deg, c)
        f = sc.amplitude_f(V, maxl, isbox=True, r0=5)
        plt.plot(theta, sc.modsq(f(theta)), label=str(maxl))

    fan = b.amplitude_f(maxl)
    plt.plot(theta, sc.modsq(fan(theta)), label='partial wave')
    plt.legend()
    plt.xlabel('hbar={}, k0={}, m={}, V={}, maxl={}, deg={}'.format(hbar, k0, m, 'box(5, -1)', maxls, deg))
    plt.show()
    
    
#sc.setquadrature('lag')
#compare(hbar=1, k0=1, m=1, r0=5, V0=-1, deg=40, c=10, maxl=10)
#sc_l_deg(hbar=1, k0=1, m=1, V=V_sample, degs=[20, 30, 40, 50, 60], c=10, maxl=10)
#sc_l_maxl(hbar=1, k0=10, m=1, V=V_sample, deg=40, c=10, maxls=[5, 10, 20, 30, 40])

sc.setquadrature('lag')
sc.setquadparams(c=0.1)
V = an.boxpot(5, -1)
V0 = sc.V_l_box(V, 0, 5)
print(V0)
V0_lag = sc.V_l(V, 0)
print(V0_lag)
print((V0_lag-V0)/V0)
