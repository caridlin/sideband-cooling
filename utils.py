#!/usr/bin/env python

from __future__ import division, print_function, unicode_literals

def cache_result(func):
    __cache = {}
    def _func(*arg):
        if arg in __cache:
            return __cache[arg]
        res = func(*arg)
        __cache[arg] = res
        return res
    return _func
