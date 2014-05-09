#!/usr/bin/env python

def cache_result(func):
    __cache = {}
    def _func(*arg):
        if arg in __cache:
            return __cache[arg]
        res = func(*arg)
        __cache[arg] = res
        return res
    return _func
