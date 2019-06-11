import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from time import time


class Particle:
    def __init__(self, dt, r0, b, v0):
        self.m = 1
        self.alpha = -1000

        r = np.sqrt(b*b + r0*r0)
        theta = np.pi-np.arctan(b/r0)
        
        self.dt = dt
        self.r = r
        self.theta = theta
        self.r_ = v0 * np.cos(theta)
        self.theta_ = -v0 * np.sin(theta) / r

    def getCoords(self):
        return [self.r * np.cos(self.theta), self.r * np.sin(self.theta)]

    def timestep(self):
        r = self.r
        r_ = self.r_
        m = self.m
        alpha = self.alpha
        th = self.theta
        th_ = self.theta_
        r__ = alpha / (m*r*r) + r * th_ * th_
        th__ = -2*r_*th_/r

        self.r_ += r__ * self.dt
        self.r += self.r_ * self.dt
        self.theta_ += th__ * self.dt
        self.theta += self.theta_ * self.dt
        

def main():
    fig = plt.figure()
    ax = plt.axes(xlim=(-50, 50), ylim=(-50, 50), aspect='equal')
    ax.grid()
    linep, = ax.plot([], [], lw=1)
    
    ptcl = Particle(0.001, 0.00001, 20, 7.1)

    x = []
    y = []
    def init():
        linep.set_data([], [])
        return linep,

    def animate(i):
        ptcl.timestep()
        coords = ptcl.getCoords()
        x.append(coords[0])
        y.append(coords[1])
        linep.set_data(x, y)
        return linep,
    t0 = time()
    animate(0)
    t1 = time()
    intv = ptcl.dt * 1000 - (t1-t0)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=300, interval=intv, blit=True)
    plt.show()

main()
