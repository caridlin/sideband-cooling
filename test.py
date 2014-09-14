#!/usr/bin/env python

from __future__ import division, print_function, unicode_literals

from sideband import sb_strength, scatter_strength
from trap import ODT
from cooling import pump_mat, raman_mat, calc_ddt_pump, calc_ddt_raman
from cooling import gamma_pump, omega_raman, evolve_rho
from ode import solve_ode
from numpy import *
import numpy as np

def _load_jsons(fnames, field):
    import json
    if isinstance(fnames, str):
        fnames = [fnames]
    res = []
    for fname in fnames:
        with open(fname) as fh:
            res.extend(json.load(fh)[field])
    return res

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
    # odt = ODT(.5e-3, 589.3e-9, 650e-9, NA=0.55)
    # print(odt)
    # print()
    # odt = ODT(10e-3, 589.3e-9, 1000e-9, NA=0.3)
    # print(odt)
    # print()
    odt = ODT(5e-3, 589.3e-9, 1000e-9, NA=0.4)
    print(odt)
    print()
    # odt = ODT(10e-3, 589.3e-9, 1000e-9, NA=0.4)
    # print(odt)
    # print()

def main_cooling():
    __import__("matplotlib").rcParams.update({'axes.labelsize': 20,
                                              'axes.titlesize': 20})
    from pylab import plot, show, imshow, figure, colorbar, xlabel, ylabel
    from pylab import legend, title, savefig, close, grid

    figure()
    def plot_pump_mat(n, eta, n0):
        plot(arange(n + 1), abs(pump_mat(n, .8, 0))[n0],
             label='$\eta=%.2f, n_0=%d$' % (eta, n0), linewidth=2,
             linestyle='-', marker='.')
    plot_pump_mat(80, .8, 0)
    plot_pump_mat(80, .8, 10)
    plot_pump_mat(80, .8, 20)
    plot_pump_mat(80, .8, 30)
    plot_pump_mat(80, .8, 40)
    plot_pump_mat(80, .8, 50)
    plot_pump_mat(80, .8, 60)
    xlabel('$n$')
    legend()
    grid()
    # title('Optical pumping branching faction\n($\\theta=0$)')
    savefig('pump_0.8_0_curve.png', bbox_inches='tight')
    close()

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
        plot(arange(dn, dn + n + 1), abs(raman_mat(n, eta, dn, 0))**2,
             label='$\eta=%.2f, \delta n=%d$' % (eta, dn), linewidth=2,
             linestyle='-', marker='.')
    plot_raman_mat(140, .8, 20)
    plot_raman_mat(140, .8, 8)
    plot_raman_mat(140, .8, 1)
    # title('Coupling ($|\\langle n|e^{ikr}|n-\\delta n\\rangle|^2$)\n'
    #       'for different $\\delta n$ and $n$')
    xlabel('$n$')
    ylabel(r'$|\langle n|e^{ikr}|n-\delta n\rangle|^2$')
    legend()
    grid()
    savefig('raman_0.8_1.png', bbox_inches='tight')
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
    # 5.244526

    n = 100
    nstart = 30
    # pumpp = 1
    pumpp = 2
    # pumpp = 4
    theta_raman = 0
    # theta_raman = -pi / 5
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

def main_raman_sb_cooling3():
    # pumpp = 1
    # theta_raman = 0
    # [8, 5, 10, 5, 5, 7, 5, 3, 7, 5, 3, 9, 6, 4, 8, 4, 2, 6, 5, 5, 3, 7, 3,
    #  7, 2, 6, 6, 4, 2, 8, 5, 3, 7, 2, 6, 5, 3, 1, 8, 6]
    # 6.302077

    # pumpp = 2
    # theta_raman = 0
    # [8, 6, 6, 6, 4, 8, 4, 7, 3, 6, 6, 4, 8, 3, 7, 5, 3, 10, 6, 3, 8, 2, 6,
    #  4, 4, 2, 8, 5, 3, 2, 6, 4, 2, 8, 4, 2, 6, 2, 5, 3]
    # 5.244526

    # pumpp = 2
    # theta_raman = -pi / 5
    # [11, 9, 7, 13, 6, 10, 5, 9, 9, 5, 11, 7, 4, 12, 10, 6, 3, 10, 5, 12, 8,
    #  4, 3, 11, 7, 5, 2, 10, 4, 12, 7, 3, 11, 5, 2, 9, 4, 3, 2, 10]
    # 4.248303

    # pumpp = 4
    # theta_raman = -pi / 5
    # [11, 9, 7, 13, 11, 8, 6, 6, 13, 10, 5, 8, 4, 12, 6, 10, 5, 8, 4, 12, 6,
    #  4, 10, 3, 8, 5, 13, 6, 4, 3, 10, 5, 3, 8, 4, 12, 6, 3, 5, 4]
    # 6.325143

    n = 100
    nstart = 30

    # pumpp = 2 * ones(n)
    theta_raman = 0
    # dns = (exp(-arange(40) * 0.028) * 7.8).astype(int) 7.112772
    # dns = (exp(-arange(40) * 0.025) * 8).astype(int) 6.451
    # dns = (exp(-arange(40) * 0.022) * 8.2).astype(int) 5.848752
    # dns = (exp(-arange(40) * 0.020) * 8.2).astype(int) 5.675294
    # dns = (exp(-arange(40) * 0.022) * 8.4).astype(int) 5.553641
    # dns = (exp(-arange(40) * 0.020) * 8.4).astype(int) 5.547241
    # dns = (exp(-arange(40) * 0.018) * 8.4).astype(int) 5.736700
    # dns = (exp(-arange(40) * 0.020) * 8.6).astype(int) 5.478592
    # dns = (exp(-arange(40) * 0.018) * 8.6).astype(int) 5.544554
    # dns = (exp(-arange(40) * 0.020) * 9).astype(int) 5.299673

    # pumpp = 1 * exp(arange(20) * .09) 3.046997
    # pumpp = 1.1 * exp(arange(20) * .08) 3.035428
    # pumpp = .9 * exp(arange(20) * .1) 3.081284
    # pumpp = .9 * exp(arange(20) * .08) 3.177622
    # pumpp = 1.1 * exp(arange(20) * .1) 3.013128
    # pumpp_name = '1.2 * exp(arange(20) * .1)' 3.039314
    # pumpp_name = '1.2 * exp(arange(20) * .12)' 3.086417
    # pumpp_name = '1.1 * exp(arange(20) * .12)' 3.028615
    # pumpp_name = '1.1 * exp(arange(20) * .1) * arange(20)**0.05' 2.865663
    # pumpp_name = '1.0 * exp(arange(20) * .1) * arange(20)**0.05' 2.860178
    # pumpp_name = '0.9 * exp(arange(20) * .1) * arange(20)**0.05' 2.899644
    # pumpp_name = '1.1 * exp(arange(20) * .1) * arange(20)**0.1' 3.040471
    # pumpp_name = '1. * exp(arange(20) * .1) * arange(20)**0.1' 2.859388
    # pumpp_name = '0.9 * exp(arange(20) * .1) * arange(20)**0.1' 2.869879
    # pumpp_name = '1.05 * exp(arange(20) * .1) * arange(20)**0.15' 2.905376
    # pumpp_name = '1. * exp(arange(20) * .1) * arange(20)**0.15' 2.880614
    # pumpp_name = '1.05 * exp(arange(20) * .1) * arange(20)**0.1' 2.870642
    # pumpp_name = '1.1 * exp(arange(20) * .1) * arange(20)**0.15' 2.938827
    # pumpp_name = '1. * exp(arange(20) * .1) * arange(20)**0.15'
    # pumpp_name = '1.1 * exp(arange(20) * .1) * arange(20)**0.1' 2.859388
    # pumpp_name = '.9 * exp(arange(20) * .1) * arange(20)**0.15' 2.862172
    # pumpp_name = '.95 * exp(arange(20) * .1) * arange(20)**0.15' 2.865720
    # pumpp_name = '.9 * exp(arange(20) * .1) * arange(20)**0.1' 2.869879
    # pumpp_name = '.95 * exp(arange(20) * .1) * arange(20)**0.1' 2.858590
    # pumpp_name = '.95 * exp(arange(20) * .1) * arange(20)**0.1' 2.391270
    pumpp_name = '.95 * exp(arange(20) * .1) * arange(20)**0.1'
    # theta_raman = 0
    # dns = (exp(-arange(80) * 0.020) * 9).astype(int) 3.325853
    # dns = (exp(-arange(80) * 0.020) * 9.5).astype(int) 2.589682
    # dns = (exp(-arange(80) * 0.020) * 10).astype(int) 2.391270
    # dns = (exp(-arange(80) * 0.020) * 11).astype(int) 1.992798
    # dns = (exp(-arange(80) * 0.020) * 12).astype(int) 1.742741
    dns_name = '(exp(-arange(80) * 0.020) * 13).astype(int)' # 1.670308
    # dns_name = '(exp(-arange(80) * 0.020) * 14).astype(int)' 1.727176
    # dns_name = '(exp(-arange(80) * 0.020) * 15).astype(int)' 2.004741
    pumpp = eval(pumpp_name)

    # pumpp = 2 * ones(n)
    # theta_raman = -pi / 5
    # dns = exp(2.48 - arange(40) * 0.028).astype(int) 4.728015
    # dns = exp(2.50 - arange(40) * 0.030).astype(int) 4.500523
    # dns = exp(2.46 - arange(40) * 0.030).astype(int) 4.839641
    # dns = exp(2.46 - arange(40) * 0.026).astype(int) 4.752056
    # dns = exp(2.50 - arange(40) * 0.026).astype(int) 4.387730

    # pumpp = 4 * ones(n)
    # theta_raman = -pi / 5
    # dns = exp(2.48 - arange(40) * 0.028).astype(int) 7.151005
    # dns = exp(2.5 - arange(40) * 0.025).astype(int) 6.577263
    # dns = exp(2.52 - arange(40) * 0.022).astype(int) 6.347822

    dns = eval(dns_name)

    ps0 = (exp(-arange(n + 1, dtype=complex128) / nstart) *
           (1 - exp(-1 / nstart)))
    rho0 = diag(r_[ps0, zeros(n + 1, dtype=complex128)])
    rho_t = [rho0]

    for i, dn in enumerate(dns):
        print("iteration: %d, dn: %d" % (i, dn))
        number = abs(sum(diag(rho0)))
        ntotal_init = calc_total_n(rho0)
        vmax = number**2 / ntotal_init
        print("atom number: %f" % number)
        print("total n: %f" % ntotal_init)
        print("v: %f" % vmax)
        ts, rhos = evolve_rho(rho0, 2, 0.1, 0.8, dn, 0.4, 0,
                              theta_raman, pi / 2, pumpp[dn - 1], 0.05)
        number = abs(sum(diag(rhos[-1])))
        ntotal = calc_total_n(rhos[-1])
        v = number**2 / ntotal
        print("atom number: %f" % number)
        print("total n: %f" % ntotal)
        print("v: %f" % v)
        print("n decreases: %f" % (ntotal_init - ntotal))
        print('')
        rho0 = rhos[-1]
        rho_t.append(rho0)
        print('\n')

    print(dns_name)
    print(pumpp_name)
    # for i, rho in enumerate(rho_t):
    #     print(i, rho)
    # for i, rho in enumerate(rho_t):
    #     print(i, diag(rho))
    rho_t = array(rho_t)
    with open('res9.json', 'w') as fh:
        import json
        json.dump({'dns_name': dns_name,
                   'pumpp_name': pumpp_name,
                   'ps': [abs(diag(rho)).tolist() for rho in rho_t],
                   'ns': [calc_total_n(rho) for rho in rho_t],
                   'rho0.real': rho0.real.tolist(),
                   'rho0.imag': rho0.imag.tolist()}, fh)

def main_raman_sb_cooling4():
    theta_raman = 0

    pumpp_name = '.65 * exp(arange(20) * .1) * arange(20)**0.12'
    dns_name = '(ones(100) * 3).astype(int)'
    pumpp = eval(pumpp_name)
    dns = eval(dns_name)

    with open('res9.json', 'r') as fh:
        import json
        d = json.load(fh)
    rho0 = array(d['rho0.real']) + 1j * array(d['rho0.imag'])
    rho_t = [rho0]

    for i, dn in enumerate(dns):
        print("iteration: %d, dn: %d" % (i, dn))
        number = abs(sum(diag(rho0)))
        ntotal_init = calc_total_n(rho0)
        vmax = number**2 / ntotal_init
        print("atom number: %f" % number)
        print("total n: %f" % ntotal_init)
        print("v: %f" % vmax)
        ts, rhos = evolve_rho(rho0, 2, 0.1, 0.8, dn, 0.4, 0,
                              theta_raman, pi / 2, pumpp[dn - 1], 0.05)
        number = abs(sum(diag(rhos[-1])))
        ntotal = calc_total_n(rhos[-1])
        v = number**2 / ntotal
        print("atom number: %f" % number)
        print("total n: %f" % ntotal)
        print("v: %f" % v)
        print("n decreases: %f" % (ntotal_init - ntotal))
        print('')
        rho0 = rhos[-1]
        rho_t.append(rho0)
        print('\n')

    print(dns_name)
    print(pumpp_name)
    # for i, rho in enumerate(rho_t):
    #     print(i, rho)
    # for i, rho in enumerate(rho_t):
    #     print(i, diag(rho))
    rho_t = array(rho_t)
    with open('res24.json', 'w') as fh:
        import json
        json.dump({'dns_name': dns_name,
                   'pumpp_name': pumpp_name,
                   'ps': [abs(diag(rho)).tolist() for rho in rho_t],
                   'ns': [calc_total_n(rho) for rho in rho_t],
                   'rho0.real': rho0.real.tolist(),
                   'rho0.imag': rho0.imag.tolist()}, fh)

def main_raman_sb_cooling5():
    theta_raman = 0

    pumpp_name = '.3 * exp(arange(20) * .1) * arange(20)**0.12'
    dns_name = '(ones(100) * 2).astype(int)'
    pumpp = eval(pumpp_name)
    dns = eval(dns_name)

    with open('res24.json', 'r') as fh:
        import json
        d = json.load(fh)
    rho0 = array(d['rho0.real']) + 1j * array(d['rho0.imag'])
    rho_t = [rho0]

    for i, dn in enumerate(dns):
        print("iteration: %d, dn: %d" % (i, dn))
        number = abs(sum(diag(rho0)))
        ntotal_init = calc_total_n(rho0)
        vmax = number**2 / ntotal_init
        print("atom number: %f" % number)
        print("total n: %f" % ntotal_init)
        print("v: %f" % vmax)
        ts, rhos = evolve_rho(rho0, 2, 0.1, 0.8, dn, 0.4, 0,
                              theta_raman, pi / 2, pumpp[dn - 1], 0.05)
        number = abs(sum(diag(rhos[-1])))
        ntotal = calc_total_n(rhos[-1])
        v = number**2 / ntotal
        print("atom number: %f" % number)
        print("total n: %f" % ntotal)
        print("v: %f" % v)
        print("n decreases: %f" % (ntotal_init - ntotal))
        print('')
        rho0 = rhos[-1]
        rho_t.append(rho0)
        print('\n')

    print(dns_name)
    print(pumpp_name)
    # for i, rho in enumerate(rho_t):
    #     print(i, rho)
    # for i, rho in enumerate(rho_t):
    #     print(i, diag(rho))
    rho_t = array(rho_t)
    with open('res26.json', 'w') as fh:
        import json
        json.dump({'dns_name': dns_name,
                   'pumpp_name': pumpp_name,
                   'ps': [abs(diag(rho)).tolist() for rho in rho_t],
                   'ns': [calc_total_n(rho) for rho in rho_t],
                   'rho0.real': rho0.real.tolist(),
                   'rho0.imag': rho0.imag.tolist()}, fh)

def main_raman_sb_cooling6():
    theta_raman = 0

    pumpp_name = '[.2]'
    dns_name = '(ones(100)).astype(int)'
    pumpp = eval(pumpp_name)
    dns = eval(dns_name)

    with open('res26.json', 'r') as fh:
        import json
        d = json.load(fh)
    rho0 = array(d['rho0.real']) + 1j * array(d['rho0.imag'])
    rho_t = [rho0]

    for i, dn in enumerate(dns):
        print("iteration: %d, dn: %d" % (i, dn))
        number = abs(sum(diag(rho0)))
        ntotal_init = calc_total_n(rho0)
        vmax = number**2 / ntotal_init
        print("atom number: %f" % number)
        print("total n: %f" % ntotal_init)
        print("v: %f" % vmax)
        ts, rhos = evolve_rho(rho0, 2, 0.1, 0.8, dn, 0.4, 0,
                              theta_raman, pi / 2, pumpp[dn - 1], 0.01)
        # ts, rhos = evolve_rho(rho0, 2, 0.1, 0.8, dn, 0.4, 0,
        #                       theta_raman, pi / 2, pumpp[dn - 1], 0.05)
        number = abs(sum(diag(rhos[-1])))
        ntotal = calc_total_n(rhos[-1])
        v = number**2 / ntotal
        print("atom number: %f" % number)
        print("total n: %f" % ntotal)
        print("v: %f" % v)
        print("n decreases: %f" % (ntotal_init - ntotal))
        print('')
        rho0 = rhos[-1]
        rho_t.append(rho0)
        print('\n')

    print(dns_name)
    print(pumpp_name)
    # for i, rho in enumerate(rho_t):
    #     print(i, rho)
    # for i, rho in enumerate(rho_t):
    #     print(i, diag(rho))
    rho_t = array(rho_t)
    with open('res29.json', 'w') as fh:
        import json
        json.dump({'dns_name': dns_name,
                   'pumpp_name': pumpp_name,
                   'ps': [abs(diag(rho)).tolist() for rho in rho_t],
                   'ns': [calc_total_n(rho) for rho in rho_t],
                   'rho0.real': rho0.real.tolist(),
                   'rho0.imag': rho0.imag.tolist()}, fh)

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

def main_plot():
    __import__("matplotlib").rcParams.update({'axes.labelsize': 20,
                                              'axes.titlesize': 20})
    from pylab import plot, show, imshow, figure, colorbar, xlabel, ylabel
    from pylab import legend, title, savefig, close, grid
    import json
    with open('res5.json') as fh:
        res = json.load(fh)

    figure()
    for i, p in enumerate(res['ps']):
        if (i + 1) % 20 == 0 or i == 0:
            p1 = (array(p[:len(p) // 2]) + p[len(p) // 2:])[:30]
            plot(p1, label="$t = %d$" % i, linewidth=2,
                 linestyle='-', marker='.')
    xlabel('$n$')
    legend()
    grid()
    # title("Energy level distribution\nat different time.")
    savefig('cool_process.png', bbox_inches='tight')

    figure()
    plot(res['ns'], linewidth=2, linestyle='-', marker='.')
    grid()
    xlabel('$t$')
    ylabel('$n$')
    # title("Average $n$ as a function of time.")
    savefig('n_decrease.png', bbox_inches='tight')
    # show()

def main_plot2():
    __import__("matplotlib").rcParams.update({'axes.labelsize': 20,
                                              'axes.titlesize': 20})
    from pylab import plot, show, imshow, figure, colorbar, xlabel, ylabel
    from pylab import legend, title, savefig, close, grid
    names = ['res5.json', 'res24.json', 'res26.json', 'res29.json']
    ps = _load_jsons(names, 'ps')
    ns = _load_jsons(names, 'ns')

    figure()
    for i in r_[0:len(ps) - 1:7j]:
        i = int(i)
        p = ps[i]
        p1 = (array(p[:len(p) // 2]) + p[len(p) // 2:])[:10]
        plot(p1, label="$t = %d$" % i, linewidth=2,
             linestyle='-', marker='.')
    xlabel('$n$')
    legend()
    grid()
    title("Energy level distribution\nat different time.")
    savefig('cool_process85.png', bbox_inches='tight')

    figure()
    plot(ns, linewidth=2, linestyle='-', marker='.')
    grid()
    xlabel('$t$')
    ylabel('$n$')
    title("Average $n$ as a function of time.")
    savefig('n_decrease85.png', bbox_inches='tight')
    show()

def main_plot3():
    __import__("matplotlib").rcParams.update({'axes.labelsize': 20,
                                              'axes.titlesize': 20})
    from pylab import plot, show, imshow, figure, colorbar, xlabel, ylabel
    from pylab import legend, title, savefig, close, grid
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm

    names = ['res5.json', 'res24.json', 'res26.json', 'res29.json']
    ps = _load_jsons(names, 'ps')
    ns = _load_jsons(names, 'ns')

    fig = figure()
    ax = fig.gca(projection='3d')
    T_fine = r_[0:len(ps) - 1]
    N_fine = r_[:10]
    T_fine, N_fine = np.meshgrid(T_fine, N_fine)

    P_all = array([(array(p[:len(p) // 2]) + p[len(p) // 2:]) for p in ps])
    P_fine = P_all[T_fine, N_fine]
    surf = ax.plot_surface(P_fine, N_fine, T_fine, rstride=1, cstride=1,
                           cmap=cm.jet_r, linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    show()

    # for i in r_[0:len(ps) - 1:7j]:
    #     i = int(i)
    #     p = ps[i]
    #     p1 = (array(p[:len(p) // 2]) + p[len(p) // 2:])[:10]
    #     plot(p1, label="$t = %d$" % i, linewidth=2,
    #          linestyle='-', marker='.')
    # xlabel('$n$')
    # legend()
    # grid()
    # title("Energy level distribution\nat different time.")
    # # savefig('cool_process85.png', bbox_inches='tight')

def main_plot3():
    __import__("matplotlib").rcParams.update({'axes.labelsize': 20,
                                              'axes.titlesize': 20})
    from pylab import plot, show, imshow, figure, colorbar, xlabel, ylabel
    from pylab import legend, title, savefig, close, grid
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mayavi import mlab

    names = ['res5.json', 'res24.json', 'res26.json', 'res28.json']
    ps = _load_jsons(names, 'ps')
    ns = _load_jsons(names, 'ns')
    print(len(ps[0]))

    T_fine = r_[0:len(ps) - 1]
    N_fine = r_[:101]
    T_fine, N_fine = np.meshgrid(T_fine, N_fine)

    P_all = array([(array(p[:len(p) // 2]) + p[len(p) // 2:]) for p in ps])
    P_fine = P_all[T_fine, N_fine]
    surf = mlab.mesh(N_fine / (len(N_fine) - 1), P_fine, T_fine / (len(ps) - 1),
                     colormap='blue-red')
    surf.module_manager.scalar_lut_manager.reverse_lut = True

    ax = mlab.axes(xlabel='State (n)', ylabel="Population", zlabel="Time",
                   nb_labels=6, extent=[0, 1, 0, 1, 0, 1],
                   ranges=[0, len(N_fine) - 1, 0, 1, 0, 1])
    ax.label_text_property.font_size = 5

    mlab.outline(surf, color=(.7, .7, .7),
                 extent=[0, 1, 0, 1, 0, 1])
    mlab.show()

    # for i in r_[0:len(ps) - 1:7j]:
    #     i = int(i)
    #     p = ps[i]
    #     p1 = (array(p[:len(p) // 2]) + p[len(p) // 2:])[:10]
    #     plot(p1, label="$t = %d$" % i, linewidth=2,
    #          linestyle='-', marker='.')
    # xlabel('$n$')
    # legend()
    # grid()
    # title("Energy level distribution\nat different time.")
    # # savefig('cool_process85.png', bbox_inches='tight')

def new_axis(fig=None):
    from mayavi.modules.axes import Axes
    from mayavi.tools.engine_manager import get_engine, engine_manager
    from mayavi.tools.figure import gcf
    scene = gcf()
    if fig is None:
        engine = get_engine()
    else:
        engine = engine_manager.find_figure_engine(fig)
        engine.current_scene = fig

    if scene.scene is not None:
        scene.scene.disable_render = True
    parent = engine.current_object

    # Try to find an existing module, if not add one to the pipeline
    if parent == None:
        target = scene
    else:
        target = parent

    ax = Axes()

    engine.add_module(ax, obj=parent)

    if scene.scene is not None:
        scene.scene.disable_render = False
    return ax

def main_plot4():
    __import__("matplotlib").rcParams.update({'axes.labelsize': 20,
                                              'axes.titlesize': 20})
    from mayavi import mlab

    names = ['res5.json', 'res24.json', 'res26.json', 'res29.json']
    ps = _load_jsons(names, 'ps')
    ns = _load_jsons(names, 'ns')
    print(len(ps[0]))

    size = 100
    t_size = 100

    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

    T_fine = r_[0:len(ps) - 1:(t_size + 1) * 1j].astype(int)
    N_fine = r_[:size + 1]
    T_fine, N_fine = np.meshgrid(T_fine, N_fine)

    P_all = array([(array(p[:len(p) // 2]) + p[len(p) // 2:]) for p in ps])
    P_fine = P_all[T_fine, N_fine]
    surf = mlab.barchart(P_fine * size, colormap='blue-red',
                         extent=[0, size, 0, size, 0, 1], figure=fig)
    # surf.module_manager.scalar_lut_manager.reverse_lut = True

    ax0 = new_axis(fig)
    ax0.axes.x_label = "State (n)"
    ax0.axes.y_label = "Time"
    ax0.axes.z_label = "Population"
    ax0.axes.x_axis_visibility = True
    ax0.axes.y_axis_visibility = False
    ax0.axes.z_axis_visibility = False
    ax0.axes.number_of_labels = 6
    ax0.axes.use_data_bounds = False
    ax0.axes.bounds = [0, size, 0, size, 0, size]
    ax0.axes.ranges = [0, len(N_fine) - 1, 0, 1, 0, 1]
    ax0.axes.use_ranges = True
    ax0.label_text_property.font_size = 3
    ax0.axes.label_format = '%.1f'

    ax1 = new_axis(fig)
    ax1.axes.x_label = "State (n)"
    ax1.axes.y_label = "Time"
    ax1.axes.z_label = "Population"
    ax1.axes.x_axis_visibility = False
    ax1.axes.y_axis_visibility = True
    ax1.axes.z_axis_visibility = False
    ax1.axes.number_of_labels = 6
    ax1.axes.use_data_bounds = False
    ax1.axes.bounds = [0, size, 0, size, 0, size]
    ax1.axes.ranges = [0, len(N_fine) - 1, 0, 1, 0, 1]
    ax1.axes.use_ranges = True
    ax1.label_text_property.font_size = 3
    ax1.axes.label_format = '%.1f'

    ax2 = new_axis(fig)
    ax2.axes.x_label = "State (n)"
    ax2.axes.y_label = "Time"
    ax2.axes.z_label = "Population"
    ax2.axes.x_axis_visibility = True
    ax2.axes.y_axis_visibility = False
    ax2.axes.z_axis_visibility = False
    ax2.axes.number_of_labels = 6
    ax2.axes.use_data_bounds = False
    ax2.axes.bounds = [0, size, 0, size, 0, size]
    ax2.axes.ranges = [len(N_fine) - 1, 0, 0, 1, 0, 1]
    ax2.axes.use_ranges = True
    ax2.label_text_property.font_size = 3
    ax2.axes.label_format = '%.0f'
    ax2.axes.fly_mode = 'none'
    ax2.axes.corner_offset = 2

    mlab.outline(surf, color=(0., 0., 0.),
                 extent=[0, size, 0, size, 0, size])
    # mlab.show_pipeline()
    mlab.view(0, 90)
    mlab.show()

def main_animate():
    __import__("matplotlib").rcParams.update({'axes.labelsize': 20,
                                              'axes.titlesize': 20})
    from pylab import plot, show, imshow, figure, colorbar, xlabel, ylabel
    from pylab import legend, title, savefig, close, grid, xlim, ylim
    from matplotlib import animation
    import json
    with open('res5.json') as fh:
        res = json.load(fh)
    ps = array([array(p[:len(p) // 2]) + p[len(p) // 2:] for p in res['ps']])

    fig = figure()
    line, = plot([], [], linewidth=2, linestyle='-', marker='.')
    xlim(0, len(ps[0]))
    ylim(0, ps.max())

    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        x = arange(len(ps[i]))
        y = ps[i]
        line.set_data(x, y)
        return line,
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(ps), interval=50, blit=True)
    grid()
    title("Energy level distribution evolution")
    anim.save('cooling.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
    show()

def main_animate2():
    __import__("matplotlib").rcParams.update({'axes.labelsize': 20,
                                              'axes.titlesize': 20})
    from pylab import plot, show, imshow, figure, colorbar, xlabel, ylabel
    from pylab import legend, title, savefig, close, grid, xlim, ylim
    from matplotlib import animation
    import json
    with open('res5.json') as fh:
        res1 = json.load(fh)
    with open('res24.json') as fh:
        res2 = json.load(fh)
    ps1 = array([array(p[:len(p) // 2]) + p[len(p) // 2:] for p in res1['ps']])
    ps2 = array([array(p[:len(p) // 2]) + p[len(p) // 2:] for p in res2['ps']])
    ps = r_[ps1, ps2]

    fig = figure()
    line, = plot([], [], linewidth=2, linestyle='-', marker='.')
    xlim(0, len(ps[0]))
    ylim(0, ps.max())

    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        x = arange(len(ps[i]))
        y = ps[i]
        line.set_data(x, y)
        return line,
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(ps), interval=50, blit=True)
    grid()
    title("Energy level distribution evolution")
    # anim.save('cooling2.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
    show()

def main_animate3():
    __import__("matplotlib").rcParams.update({'axes.labelsize': 20,
                                              'axes.titlesize': 20})
    from pylab import plot, show, imshow, figure, colorbar, xlabel, ylabel
    from pylab import legend, title, savefig, close, grid, xlim, ylim
    from matplotlib import animation
    import json
    with open('res5.json') as fh:
        res1 = json.load(fh)
    with open('res24.json') as fh:
        res2 = json.load(fh)
    with open('res26.json') as fh:
        res3 = json.load(fh)
    ps1 = array([array(p[:len(p) // 2]) + p[len(p) // 2:] for p in res1['ps']])
    ps2 = array([array(p[:len(p) // 2]) + p[len(p) // 2:] for p in res2['ps']])
    ps3 = array([array(p[:len(p) // 2]) + p[len(p) // 2:] for p in res3['ps']])
    ps = r_[ps1, ps2, ps3]

    fig = figure()
    line, = plot([], [], linewidth=2, linestyle='-', marker='.')
    xlim(0, len(ps[0]))
    ylim(0, ps.max())

    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        x = arange(len(ps[i]))
        y = ps[i]
        line.set_data(x, y)
        return line,
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(ps), interval=50, blit=True)
    grid()
    title("Energy level distribution evolution")
    # anim.save('cooling2.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
    show()

def main_animate4():
    __import__("matplotlib").rcParams.update({'axes.labelsize': 20,
                                              'axes.titlesize': 20})
    from pylab import plot, show, imshow, figure, colorbar, xlabel, ylabel
    from pylab import legend, title, savefig, close, grid, xlim, ylim
    from matplotlib import animation
    import json
    with open('res5.json') as fh:
        res1 = json.load(fh)
    with open('res24.json') as fh:
        res2 = json.load(fh)
    with open('res26.json') as fh:
        res3 = json.load(fh)
    with open('res29.json') as fh:
        res4 = json.load(fh)
    ps1 = array([array(p[:len(p) // 2]) + p[len(p) // 2:] for p in res1['ps']])
    ps2 = array([array(p[:len(p) // 2]) + p[len(p) // 2:] for p in res2['ps']])
    ps3 = array([array(p[:len(p) // 2]) + p[len(p) // 2:] for p in res3['ps']])
    ps4 = array([array(p[:len(p) // 2]) + p[len(p) // 2:] for p in res4['ps']])
    ps = r_[ps1, ps2, ps3, ps4]

    fig = figure()
    line, = plot([], [], linewidth=2, linestyle='-', marker='.')
    xlim(0, len(ps[0]))
    ylim(0, ps.max())

    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        x = arange(len(ps[i]))
        y = ps[i]
        line.set_data(x, y)
        return line,
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(ps), interval=50, blit=True)
    grid()
    title("Energy level distribution evolution")
    # anim.save('cooling4.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
    show()

def main():
    # main_sideband()
    # main_odt()
    # main_cooling()
    # main_ode()
    # main_pump()
    # main_raman_sb_cooling()
    # main_raman_sb_cooling3()
    # main_raman_sb_cooling4()
    # main_raman_sb_cooling5()
    # main_raman_sb_cooling6()
    # main_plot()
    # main_plot2()
    # main_plot3()
    main_plot4()
    # main_animate()
    # main_animate2()
    # main_animate3()
    # main_animate4()
    pass

if __name__ == '__main__':
    main()
