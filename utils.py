####Irrelevant and unimportant Utils####

import numpy as np
from matplotlib import pyplot as plt


def plotcomplex(x, f):
    u = f(x)
    plt.plot(x, np.real(u), x, np.imag(u))
    plt.show()

def plot(a, b, f):
    x = np.linspace(a, b, 10000)
    plt.plot(x, f(x))
    plt.show()

def error(x, f1x, f0x):
    return np.sqrt(sum(np.abs(f1x - f0x)**2) / sum(np.abs(f0x)**2))

def maxerror(x, f1, f0):
    return max(np.abs((f1(x)-f0(x))/f0(x))**2)
