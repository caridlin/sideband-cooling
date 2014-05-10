#!/usr/bin/env python

from __future__ import division, print_function, unicode_literals

from numpy import *

from utils import *
from constants import *
from trap import ODT
from sideband import sb_strength, scatter_strength

try:
    range = xrange
except:
    pass

@cache_result
def pump_mat(n, eta, theta0):
    res = empty((n + 1, n + 1), dtype=float64)
    for i in range(n + 1):
        for j in range(n + 1):
            res[i, j] = scatter_strength(i, j, eta, theta0)
    return res

@cache_result
def raman_mat(n, eta, dn=1, theta1=-pi / 2, theta2=pi / 2):
    eta = eta * abs(sin(theta1) - sin(theta2))
    res = empty(n + 1, dtype=complex128)
    for i in range(n + 1):
        res[i] = sb_strength(i, i + dn, eta)
    return res
