import scatter_central as sc
import partial_wave as an
import test as tt
import numpy as np
import potentials as col
from matplotlib import pyplot as plt
from scipy import special as spe
import utils

def a1():
    r = np.linspace(0, 10, 10000)
    th = np.linspace(0, np.pi, 10000)


    V = col.box(-1, 5)
    plt.plot(r, V(r))
    plt.show()

    g = an.general(1, 50, 1, V, 5)
    fg = g.amplitude_f(10)
    
    an.k0 = 50
    b = an.box(5, -1)
    fan = b.amplitude_f(10)

    sc.setquadparams(deg=80)
    sc.setparams(k01=50)
    f = sc.amplitude_f(V, 10)

    plt.plot(th, fg(th), th, f(th), th, fan(th))
    plt.xlabel(sc.summary())
    plt.show()

def a2():
    sc.setquadrature(quad='leg')
    f = lambda r: np.exp(-r**2)
    I = np.sqrt(np.pi)/2
    S = sum(sc.W * f(sc.K)) 
    print(S)
    print((S-I)/I)
    
    sc.setquadrature(quad='lag')
    S = sum(sc.W * f(sc.K))
    print((S-I)/I)

    sc.setquadrature(quad='mixed')
    sc.setquadparams(c=2, r0=0.2)
    S = sum(sc.W * f(sc.K))
    print((S-I)/I)

def kkk():
    r = np.linspace(0, 10, 10000)
    th = np.linspace(0, np.pi, 10000)
    V = col.box(-1, 5)
    plt.plot(r, V(r))
    plt.show()

    K_SAMPLE = [0.01, 0.1, 0.5, 1, 10, 20, 50, 100]
    
    for k0 in K_SAMPLE:
        g = an.general(1, k0, 1, V, 5)
        fg = g.amplitude_f(10)
        
        an.k0 = k0
        b = an.box(5, -1)
        fan = b.amplitude_f(10)

        sc.setquadparams(deg=40)
        sc.setparams(k01=k0)
        f = sc.amplitude_f(V, 10, isbox=True, r0=5)
        
        q = utils.maxerror(th, fg, f)
        #print('maxereror: {}'.format(q))
        plt.plot(th, fg(th), th, f(th), th, fan(th))
        plt.xlabel(sc.summary())
        plt.show()

def degdeg():
    r = np.linspace(0, 10, 10000)
    th = np.linspace(0, np.pi, 10000)
    V = col.shell(2, 1, 5)
    plt.plot(r, V(r))
    plt.show()
    
    g = an.general(1, 1, 1, V, 10)
    fg = g.amplitude_f(10)
    sigg = sc.modsq(fg(th))
    #b = an.box(1, -1)
    #fan = b.amplitude_f(10)

    plt.plot(th, sigg, label='pwa')
    #plt.plot(th, sc.modsq(fan(th)), label='pwa2')
    
    for deg1 in [20, 30, 40, 60, 80, 100]:
        sc.setquadparams(deg=deg1)
        f = sc.amplitude_f(V, 10)
        sig = sc.modsq(f(th))
        err = utils.error(th, sig, sigg)
        print('deg: {}, error: {}'.format(deg1, err))
        plt.plot(th, sig, label=str(deg1))

    plt.legend()
    plt.xlabel('theta')
    plt.ylabel('differential cross section')
    plt.show()

def maxlmaxl():
    r = np.linspace(0, 10, 10000)
    th = np.linspace(0, np.pi, 10000)
    V = col.ring(2, 1, 5)
    plt.plot(r, V(r))
    plt.show()

    g = an.general(1, 1, 1, V, 10)
    fg = g.amplitude_f(10)
    plt.plot(th, fg(th), label='pwa')
    sc.setquadparams(deg=40)
    for maxl1 in [5, 10, 15, 20, 40, 60]:
        f = sc.amplitude_f(V, maxl1)
        plt.plot(th, f(th), label=str(maxl1))
    plt.legend()
    plt.xlabel('V=ring(2, 1, 5), k=1, deg=40')
    plt.show()

degdeg()
