#!/usr/bin/env python

from __future__ import division, print_function, unicode_literals

from sideband import sb_strength, scatter_strength
from trap import ODT
from cooling import pump_mat, raman_mat, calc_ddt_pump, calc_ddt_raman
from cooling import gamma_pump, omega_raman, evolve_rho
from ode import solve_ode
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

def main_ode():
    # from pylab import plot, show, imshow, figure, colorbar, xlabel, ylabel
    # from pylab import legend, title, savefig, close, grid

    ts, ys = solve_ode(0, [0, 1], lambda t, y: array([y[1], -y[0]]), 10000, .01)
    # plot(ts, ys.T[0])
    # show()

def calc_total_n(rho):
    ps = diag(rho)
    n, = ps.shape
    return abs(ps.dot(r_[arange(n // 2), arange(n // 2)]))

def main_raman_sb_cooling():
    # from pylab import plot, show, imshow, figure, colorbar, xlabel, ylabel
    # from pylab import legend, title, savefig, close, grid

    n = 100
    nstart = 30
    gammas = gamma_pump(n, .8, pi / 2, 0.4, 0.2)
    omegas = [omega_raman(n, .8, dn, 0) for dn in range(1, n + 1)]
    def get_f(dn):
        def f(t, rho):
            return (calc_ddt_pump(rho, gammas) +
                    calc_ddt_raman(rho, omegas[dn - 1]) * 10)
        return f
    ps0 = (exp(-arange(n + 1, dtype=complex128) / nstart) *
           (1 - exp(-1 / nstart)))
    rho0 = diag(r_[ps0, zeros(n + 1, dtype=complex128)])
    dns = []

    dnrange = range(1, 40)
    for i in range(40):
        print("\nstart iteration: %d" % i)
        number = abs(sum(diag(rho0)))
        ntotal = calc_total_n(rho0)
        dnmax = 0
        vmax = number**2 / ntotal
        print("atom number: %f" % number)
        print("total n: %f" % ntotal)
        print("v: %f" % vmax)
        for dn in dnrange:
            print("dn: %d" % dn)
            ts, rhos = solve_ode(0, rho0, get_f(dn), .45, 0.015)
            print("dn: %d" % dn)
            number = abs(sum(diag(rhos[-1])))
            ntotal = calc_total_n(rhos[-1])
            v = number**2 / ntotal
            print("atom number: %f" % number)
            print("total n: %f" % ntotal)
            print("v: %f" % v)
            if v > vmax:
                print("use new dn: %d, v = %f" % (dn, v))
                dnmax = dn
                vmax = v
                new_rho0 = rhos[-1]
            # plot(abs(diag(rhos[0])), label='before')
            # plot(abs(diag(rhos[-1])), label='after')
            # legend()
            # figure()
            # plot(abs(diag(rhos[-1])) - abs(diag(rhos[0])))
            # figure()
            # imshow(abs(rhos[0]))
            # figure()
            # imshow(abs(rhos[-1]))
            # show()
            print('')
        if dnmax == 0:
            print("cooling stopped, abort")
            break
        dns.append(dnmax)
        dnrange = range(max(dnmax - 15, 1), dnmax + 16)
        rho0 = new_rho0
        print('\n')

    print("atom number: %f\n" % sum(diag(rho0)))
    print("total n: %f\n" % calc_total_n(rho0))
    print(dns)
    # plot(abs(diag(rhos[0])))
    # plot(abs(diag(rhos[-1])))
    # figure()
    # imshow(abs(rhos[0]))
    # figure()
    # imshow(abs(rhos[-1]))
    # show()

def main_raman_sb_cooling2():
    # from pylab import plot, show, imshow, figure, colorbar, xlabel, ylabel
    # from pylab import legend, title, savefig, close, grid

    # pumpp = 2
    # theta_raman = 0
    # [8, 6, 6, 6, 4, 8, 4, 7, 3, 6, 6, 4, 8, 3, 7, 5, 3, 10, 6, 3, 8, 2, 6,
    #  4, 4, 2, 8, 5, 3, 2, 6, 4, 2, 8, 4, 2, 6, 2, 5, 3]

    n = 100
    nstart = 30
    # pumpp = 1
    # pumpp = 2
    pumpp = 4
    # theta_raman = 0
    theta_raman = -pi / 5
    ps0 = (exp(-arange(n + 1, dtype=complex128) / nstart) *
           (1 - exp(-1 / nstart)))
    rho0 = diag(r_[ps0, zeros(n + 1, dtype=complex128)])
    dns = []

    dnrange = range(20, 0, -1)
    for i in range(40):
        print("start iteration: %d" % i)
        number = abs(sum(diag(rho0)))
        ntotal_init = calc_total_n(rho0)
        dnmax = 0
        vmax = number**2 / ntotal_init
        print("atom number: %f" % number)
        print("total n: %f" % ntotal_init)
        print("v: %f" % vmax)
        for dn in dnrange:
            print("iteration: %d, dn: %d" % (i, dn))
            ts, rhos = evolve_rho(rho0, 2, 0.1, 0.8, dn, 0.4, 0,
                                  theta_raman, pi / 2, pumpp, 0.05)
            number = abs(sum(diag(rhos[-1])))
            ntotal = calc_total_n(rhos[-1])
            v = number**2 / ntotal
            print("atom number: %f" % number)
            print("total n: %f" % ntotal)
            print("v: %f" % v)
            if v > vmax:
                print("use new dn: %d, v = %f, n decreases: %f" %
                      (dn, v, ntotal_init - ntotal))
                dnmax = dn
                vmax = v
                new_rho0 = rhos[-1]
            # plot(abs(diag(rhos[0])), label='before')
            # plot(abs(diag(rhos[-1])), label='after')
            # legend()
            # figure()
            # plot(abs(diag(rhos[-1])) - abs(diag(rhos[0])))
            # figure()
            # imshow(abs(rhos[0]))
            # figure()
            # imshow(abs(rhos[-1]))
            # show()
            print('')
        rho0 = new_rho0
        if dnmax == 0:
            print("cooling stopped, abort")
            break
        dns.append(dnmax)
        dnrange = range(max(dnmax - 8, 1), dnmax + 9)
        print('\n')

    print(rho0)
    print(diag(rho0))
    print("atom number: %f\n" % sum(diag(rho0)))
    print("total n: %f\n" % calc_total_n(rho0))
    print(dns)
    # plot(abs(diag(rhos[0])))
    # plot(abs(diag(rhos[-1])))
    # figure()
    # imshow(abs(rhos[0]))
    # figure()
    # imshow(abs(rhos[-1]))
    # show()

def main_pump():
    n = 100
    gammas = gamma_pump(n, .8, pi / 2, 0.4, 0.02)
    omegas = [omega_raman(n, .8, dn) for dn in range(1, 22)]
    # omegas = omega_raman(n, .8, 10)
    def f(t, rho):
        dn = int(21 - t)
        return calc_ddt_pump(rho, gammas) + calc_ddt_raman(rho, omegas[dn - 1])
    ps0 = exp(-arange(n + 1, dtype=complex128) / 10) * (1 - exp(1 / 10))
    rho0 = diag(r_[ps0, zeros(n + 1, dtype=complex128)])
    print(sum(rho0))
    print(calc_total_n(rho0))
    ts, rhos = solve_ode(0, rho0, f, 20, .01)
    print(abs(diag(rhos[0])))
    print(abs(diag(rhos[-1])))
    plot(abs(diag(rhos[0])))
    plot(abs(diag(rhos[-1])))
    figure()
    imshow(abs(rhos[0]))
    figure()
    imshow(abs(rhos[-1]))
    show()

def main():
    # main_sideband()
    # main_odt()
    # main_cooling()
    # main_ode()
    # main_pump()
    # main_raman_sb_cooling()
    main_raman_sb_cooling2()
    pass

if __name__ == '__main__':
    main()
