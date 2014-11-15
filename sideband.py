#!/usr/bin/env python

from __future__ import division, print_function, unicode_literals
from utils import cache_result, factorial, permutation
from numpy import *

@cache_result
def _exp_annihilation(n, eta):
    res = empty(n + 1, dtype=complex128)
    prev = complex128(1)
    res[n] = complex128(1)
    for i in range(n):
        prev *= eta * 1j / (i + 1) * sqrt(n - i)
        res[n - i - 1] = prev
    return res

@cache_result
def sb_strength(m, n, eta):
    eta = float64(eta)
    if m > n:
        # cache friendly
        return sb_strength(n, m, eta)
    size = m + 1
    return (_exp_annihilation(m, eta)[:size]
            .dot(_exp_annihilation(n, eta)[:size]) *
            exp(-eta**2 / 2))

def _calc_fact_term(m, n, dn, s, eta):
    denom1 = 2 * s + 1
    denom2 = denom1 + dn
    denom3 = m - 2 * s
    if denom3 > denom1:
        tmp = denom1
        denom1 = denom3
        denom3 = tmp
    if m > denom1:
        num = permutation(n, denom1) * permutation(m, denom1)
        denom = factorial(denom2)**2 * factorial(denom3)**2
    else:
        num = permutation(n, denom1)
        denom = (permutation(denom1, m) * factorial(denom2)**2 *
                 factorial(denom3)**2)
    res = sqrt(num / denom)
    return res

@cache_result
def sb_strength2(m, n, eta):
    eta = float64(eta)
    if m > n:
        # cache friendly
        return sb_strength2(n, m, eta)
    dn = n - m
    size = (m + 2) // 2
    buff = empty(size, dtype=float64)
    eta2 = eta**2
    eta4 = eta**4
    for s in range(size):
        buff[s] = (eta4**s * _calc_fact_term(m, n, dn, s, eta) *
                   ((2 * s + dn + 1) * (2 * s + 1) - eta2 * (n - dn - 2 * s)))
    return sum(buff) * exp(-eta2 / 2) * (eta * 1j)**dn

_thetas = r_[-pi / 2:pi / 2:91j]
_sum_weight = sum(cos(_thetas))
@cache_result
def scatter_strength(m0, n0, eta, theta0):
    if m0 > n0:
        # cache friendly
        return scatter_strength(n0, m0, eta, theta0)
    s0 = sin(theta0)
    return sum([(abs(sb_strength(m0, n0, eta * abs(sin(theta) - s0)))**2
                 * cos(theta))
                 for theta in _thetas]) / _sum_weight
