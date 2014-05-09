#!/usr/bin/env python

from __future__ import division

from sideband import sb_strength, scatter_strength
from trap import ODT
from numpy import *

def plot_strength(n0, dn, eta, theta0, scatter=True):
    ns = r_[max(n0 - dn, 0):n0 + dn]
    if scatter:
        strengths = array([scatter_strength(n, n0, eta, theta0) for n in ns])
    else:
        strengths = array([abs(sb_strength(n, n0, eta))**2 for n in ns])
    print(sum(strengths))
    from pylab import plot
    plot(ns, strengths, label=str((n0, dn, eta, theta0, scatter)))

def main_sideband():
    for n0 in range(0, 150, 10):
        plot_strength(n0, 150, 1.3, 0)
        plot_strength(n0, 150, 1.3, 0, False)
    from pylab import legend, grid, show
    legend()
    grid()
    show()

def main_odt():
    odt = ODT(3e-3, 589.3e-9, 1000e-9, NA=0.6)
    print(odt)
    print()
    odt = ODT(10e-3, 589.3e-9, 1000e-9, NA=0.3)
    print(odt)
    print()
    odt = ODT(5e-3, 589.3e-9, 1064e-9, NA=0.4)
    print(odt)
    print()

def main():
    # main_sideband()
    main_odt()
    pass

if __name__ == '__main__':
    main()
