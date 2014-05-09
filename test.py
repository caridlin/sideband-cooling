#!/usr/bin/env python

from sideband import sb_strength, scatter_strength
from numpy import *

def plot_strength(n0, dn, eta, theta0, scatter=True):
    ns = r_[max(n0 - dn, 0):n0 + dn]
    if scatter:
        strengths = array([scatter_strength(n, n0, eta, theta0) for n in ns])
    else:
        strengths = array([abs(sb_strength(n, n0, eta))**2 for n in ns])
    print(sum(strengths))
    # from pylab import plot
    # plot(ns, strengths, label=str((n0, dn, eta, theta0)))

def main():
    for n0 in range(0, 150, 10):
        plot_strength(n0, 150, 1, 0)
        plot_strength(n0, 150, 1.5, 0, False)
    # from pylab import legend, grid, show
    # legend()
    # grid()
    # show()

if __name__ == '__main__':
    main()
