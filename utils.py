import numpy as np


def box(r0, V0):
    return lambda r: V0 * (1 - np.heaviside(r-r0, 0))
