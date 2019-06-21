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

def error(x, f1, f0):
    dx = (x[-1] - x[0]) / (len(x) - 1)
    si = 0
    return np.sqrt(sum(np.abs((f1(x) - f0(x))/f0(x)) ** 2) * dx)

def maxerror(x, f1, f0):
    return max(np.abs((f1(x)-f0(x))/f0(x))**2)
