#!/usr/bin/env python

from __future__ import division

from constants import *
from utils import *

def cache_prop(func):
    return property(cache_result(func))

@cache_result
def omega_light(lamb):
    return 2 * pi * c / lamb

class ODT:
    def _w_from_NA(self, NA):
        return self.__lamb / pi / NA
    def __init__(self, P, lamb0, lamb, w0=None, NA=None, m_a=m_Na):
        self.__P = P
        self.__lamb = lamb
        self.__lamb0 = lamb0
        self.__omega = omega_light(lamb)
        self.__omega0 = omega_light(lamb0)
        self.__m_a = m_a
        if w0 is None:
            w0 = self._w_from_NA(NA)
        self.__w0 = w0
    @property
    def w0(self):
        return self.__w0
    # center intensity for gaussian beam
    @cache_prop
    def I0(self):
        return 2 * self.__P / pi / self.__w0**2
    # negetive AC stark shift
    @cache_result
    def stark_shift(self, I):
        return (I / 2 / c / epsilon_0 * E**2 / m_e /
                (self.__omega0 - self.__omega) /
                (self.__omega0 + self.__omega))
    @cache_prop
    def depth(self):
        return self.stark_shift(self.I0)
    @cache_prop
    def depth_f(self):
        return self.depth / h
    @cache_prop
    def depth_omega(self):
        return self.depth / hbar
    @cache_prop
    def rayleigh_range(self):
        return self.__w0**2 * pi / self.__lamb
    @cache_prop
    def trap_omega_l(self):
        return sqrt(self.depth / self.__m_a) / self.rayleigh_range
    @cache_prop
    def trap_omega_r(self):
        return sqrt(2 * self.depth / self.__m_a) / self.__w0
    @cache_prop
    def trap_freq_l(self):
        return self.trap_omega_l / 2 / pi
    @cache_prop
    def trap_freq_r(self):
        return self.trap_omega_r / 2 / pi
    @cache_prop
    def z0_l(self):
        return sqrt(hbar / 2 / self.__m_a / self.trap_omega_l)
    @cache_prop
    def z0_r(self):
        return sqrt(hbar / 2 / self.__m_a / self.trap_omega_r)
    @cache_prop
    def eta_l(self):
        return 2 * pi / self.__lamb0 * self.z0_l
    @cache_prop
    def eta_r(self):
        return 2 * pi / self.__lamb0 * self.z0_r
    @cache_prop
    def nmax_r(self):
        return self.depth_f / self.trap_freq_r / 2
    @cache_prop
    def nmax_l(self):
        return self.depth_f / self.trap_freq_l / 2
    def __str__(self):
        return (('ODT for <lambda0: %fnm, m: %fu>\n' %
                 (self.__lamb0 * 1e9, self.__m_a * N_A / 1e-3)) +
                 '\n'.join(['%s: %e' % (prop, getattr(self, prop))
                            for prop
                            in ('w0', 'I0', 'depth', 'depth_f', 'depth_omega',
                                'rayleigh_range', 'trap_omega_l', 'trap_omega_r',
                                'trap_freq_l', 'trap_freq_r', 'z0_l', 'z0_r',
                                'eta_l', 'eta_r', 'nmax_l', 'nmax_r')]))
