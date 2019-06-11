import numpy as np
from matplotlib import pyplot as plt

m = 1
hbar = 1
a = 1
k0 = 1
b = 1
def f1B(theta, phi):
    return -m*a/(np.pi*hbar**2) * np.cos(np.sin(theta) * np.sin(phi) * k0 * b)

def crosssec1B(theta, phi):
    f = f1B(theta, phi)
    return f * np.conjugate(f)


phi = np.linspace(0, np.pi * 2, 1000)
theta = np.linspace(0, np.pi, 1000)
PHI, THETA = np.meshgrid(phi, theta)
cross = crosssec1B(theta, PHI)

plt.plot(costheta, cross)
plt.show()
