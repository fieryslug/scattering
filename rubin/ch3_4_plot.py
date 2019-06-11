import numpy as np
from matplotlib import pyplot as plt
from scipy import special

R = 1
k = 1/R

#hard sphere
def phase(l):
    return special.spherical_jn(l, 1) / special.spherical_yn(l, 1)

def partialf(l):
    return (np.exp(2j*phase(l)) - 1) / (2j)


def f_lth(l, theta):
    return (2*l+1) * special.legendre(l)(np.cos(theta)) * partialf(l) / k

theta = np.linspace(0, np.pi, 10000)
costh = np.cos(theta)

print(phase(0))

DCS0 = np.abs(f_lth(0, theta)) ** 2
DCS1 = np.abs(f_lth(1, theta)) ** 2
DCS2 = np.abs(f_lth(2, theta)) ** 2
DCS012 = np.abs(f_lth(0, theta) + f_lth(1, theta) + f_lth(2, theta)) ** 2


plt.plot(costh, DCS2)
#plt.plot(costh, DCS012)
plt.show()
