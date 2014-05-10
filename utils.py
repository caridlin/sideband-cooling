#!/usr/bin/env python

from __future__ import division, print_function, unicode_literals
from numpy import *

def cache_result(func):
    __cache = {}
    def _func(*arg):
        if arg in __cache:
            return __cache[arg]
        res = func(*arg)
        __cache[arg] = res
        return res
    return _func

@cache_result
def factorial(n):
    n = long(n)
    if n <= 1:
        return 1L
    if n >= 16:
        factorial(n // 2)
    return n * factorial(n - 1)

@cache_result
def permutation(n, m):
    n = long(n)
    m = long(m)
    if n <= m:
        return 1L
    if n - m >= 16:
        permutation((n + m) // 2, m)
    return n * permutation(n - 1, m)
