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
