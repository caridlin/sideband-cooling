#!/usr/bin/env python

from __future__ import division, print_function, unicode_literals

from numpy import *

from utils import *
from constants import *
from trap import ODT
from sideband import sb_strength, scatter_strength
from ode import solve_ode

try:
    range = xrange
except:
    pass

#                j
#  [[(0, 0), (0, 1), (0, 2), ...],
#   [(1, 0), (1, 1), (1, 2), ...],
# i [(2, 0), (2, 1), (2, 2), ...],
#   ...]

@cache_result
def pump_mat(n, eta, theta0):
    res = empty((n + 1, n + 1), dtype=float64)
    for i in range(n + 1):
        for j in range(n + 1):
            res[i, j] = scatter_strength(i, j, eta, theta0)
    return res

@cache_result
def gamma_pump(n, eta, theta0, branch, miss=0.0):
    pump = pump_mat(n, eta, theta0)
    res = empty(((n + 1) * 2, (n + 1) * 2), dtype=float64)
    res[:n + 1, :n + 1] = pump * (miss * branch)
    res[n + 1:, :n + 1] = pump * (miss * (1 - branch))
    res[:n + 1, n + 1:] = pump * ((1 - miss) * branch)
    res[n + 1:, n + 1:] = pump * ((1 - miss) * (1 - branch))
    return res

@cache_result
def raman_mat(n, eta, dn=1, theta1=-pi / 2, theta2=pi / 2):
    eta = eta * abs(sin(theta1) - sin(theta2))
    res = empty(n + 1, dtype=complex128)
    for i in range(n + 1):
        res[i] = sb_strength(i, i + dn, eta)
    return res

@cache_result
def omega_raman(n, eta, dn=1, theta1=-pi / 2, theta2=pi / 2):
    raman = raman_mat(n, eta, dn, theta1, theta2)
    n = n + 1
    res = zeros((n * 2, n * 2), dtype=complex128)
    for i in range(n - dn):
        res[i + dn, n + i] = raman[i]
        res[n + i, i + dn] = raman[i].conjugate()
    return res

@cache_result
def norm_omega_raman(n, eta, dn=1, theta1=-pi / 2, theta2=pi / 2):
    raman = raman_mat(n, eta, dn, theta1, theta2)
    n = n + 1
    res = zeros((n * 2, n * 2), dtype=complex128)
    omega_m = max(abs(raman[:n - dn]).max(), 0.05)
    print('omega:', 1 / omega_m)
    for i in range(n - dn):
        res[i + dn, n + i] = raman[i] / omega_m
        res[n + i, i + dn] = raman[i].conjugate() / omega_m
    return res

def _upper_iter(size):
    for j in range(size):
        for i in range(j + 1):
            yield i, j

def _lower_iter(size):
    for i in range(1, size):
        for j in range(i):
            yield i, j

def calc_ddt_pump(rho, gammas):
    m, n = rho.shape
    res = empty(rho.shape, dtype=complex128)
    ps = diag(rho)
    for i, j in _upper_iter(m):
        if i == j:
            res[i, i] = ps.dot(gammas[i]) - rho[i, i] * gammas[:, i].sum()
        else:
            res[i, j] = -(gammas[:, i].sum() + gammas[:, j].sum()) / 2 * rho[i, j]
    for i, j in _lower_iter(m):
        res[i, j] = res[j, i].conjugate()
    return res

def calc_ddt_raman(rho, omegas):
    m, n = rho.shape
    return 1j * (rho.dot(omegas) - omegas.dot(rho))

def evolve_rho(rho, t, dt, eta, dn, branch, theta0,
               theta1=-pi / 2, theta2=pi / 2, pumpp=1., miss=0.0):
    m, n = rho.shape
    n = n // 2 - 1
    gammas = gamma_pump(n, eta, theta0, branch, miss)
    omegas = norm_omega_raman(n, eta, dn, theta1, theta2)
    def f(t, rho):
        return (calc_ddt_pump(rho, gammas) * pumpp +
                calc_ddt_raman(rho, omegas))
    return solve_ode(0, rho, f, t, dt)
