#!/usr/bin/env python

from __future__ import division, print_function, unicode_literals

from numpy import *

def solve_ode(t0, y0, f, t1, h):
    y0 = array(y0)
    if t1 <= t0:
        raise ValueError
    ts = r_[t0:t1:h]
    nstep = len(ts)
    ys = empty((nstep,) + y0.shape, dtype=y0.dtype)
    ys[0] = y0
    h2 = h / 2
    h3 = h / 3
    h6 = h / 6
    for i, t in enumerate(ts):
        if not i:
            continue
        prev = ys[i - 1]
        k1 = f(t, prev)
        k2 = f(t + h2, prev + h2 * k1)
        k3 = f(t + h2, prev + h2 * k2)
        k4 = f(t + h, prev + h * k3)
        ys[i] = prev + (h6 * k1 + h3 * k2 + h3 * k3 + h6 * k4)
    return ts, ys
