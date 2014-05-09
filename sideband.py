#!/usr/bin/env python

from utils import cache_result
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
    size = min(m, n) + 1
    return (_exp_annihilation(m, eta)[:size]
            .dot(_exp_annihilation(n, eta)[:size]) *
            exp(-complex128(eta)**2 / 2))

_thetas = r_[-pi / 2:pi / 2:91j]
_sum_weight = sum(cos(_thetas))
@cache_result
def scatter_strength(m0, n0, eta, theta0):
    s0 = sin(theta0)
    return sum([(abs(sb_strength(m0, n0, eta * abs(sin(theta) - s0)))**2
                 * cos(theta))
                 for theta in _thetas]) / _sum_weight
