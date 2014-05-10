#!/usr/bin/env python

from __future__ import division, print_function, unicode_literals

from sideband import sb_strength, scatter_strength
from trap import ODT
from cooling import pump_mat, raman_mat
from numpy import *

def plot_strength(n0, dn, eta, theta0, scatter=True):
    ns = r_[max(n0 - dn, 0):n0 + dn]
    if scatter:
        strengths = array([scatter_strength(n, n0, eta, theta0) for n in ns])
    else:
        strengths = array([abs(sb_strength(n, n0, eta))**2 for n in ns])
    print(n0, sum(strengths))
    from pylab import plot
    plot(ns, strengths, label=str((n0, dn, eta, theta0, scatter)))

def main_sideband():
    # sb_strength(140, 140, 2)
    # plot_strength(140, 150, 2, 0, False)
    for n0 in range(0, 151, 10):
        plot_strength(n0, 150, 1.3, 0)
        plot_strength(n0, 150, 1.6, 0, False)
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
    odt = ODT(5e-3, 589.3e-9, 1000e-9, NA=0.4)
    print(odt)
    print()
    odt = ODT(10e-3, 589.3e-9, 1000e-9, NA=0.4)
    print(odt)
    print()

def main_cooling():
    __import__("matplotlib").rcParams.update({'axes.labelsize': 20,
                                              'axes.titlesize': 20})
    from pylab import plot, show, imshow, figure, colorbar, xlabel, ylabel
    from pylab import legend, title, savefig, close, grid

    figure()
    imshow(abs(pump_mat(50, .8, 0)), origin='lower')
    xlabel('$n_1$')
    ylabel('$n_2$')
    title('Optical pumping branching faction\n($\\eta=0.8, \\theta=0$)')
    colorbar()
    savefig('pump_0.8_0.png', bbox_inches='tight')
    close()

    figure()
    imshow(abs(pump_mat(50, .8, pi / 2)), origin='lower')
    xlabel('$n_1$')
    ylabel('$n_2$')
    title('Optical pumping branching faction\n'
          '($\\eta=0.8, \\theta=\\frac{\\pi}{2}$)')
    colorbar()
    savefig('pump_0.8_pi_2.png', bbox_inches='tight')
    close()

    figure()
    def plot_raman_mat(n, eta, dn):
        plot(arange(dn, dn + n + 1), abs(raman_mat(n, eta, dn))**2,
             label='$\eta=%.2f, \delta n=%d$' % (eta * 2, dn), linewidth=2,
             linestyle='-', marker='.')
    plot_raman_mat(140, .8, 20)
    plot_raman_mat(140, .8, 8)
    plot_raman_mat(140, .8, 1)
    title('Coupling ($|\\langle n|e^{ikr}|n-\\delta n\\rangle|^2$)\n'
          'for different $\\delta n$ and $n$')
    xlabel('$n$')
    ylabel(r'$|\langle n|e^{ikr}|n-\delta n\rangle|^2$')
    legend()
    grid()
    savefig('raman_0.8.png', bbox_inches='tight')
    close()
    # show()

def main():
    # main_sideband()
    # main_odt()
    main_cooling()
    pass

if __name__ == '__main__':
    main()
