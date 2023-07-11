# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)

from numpy import arange, abs, sign, roll, frompyfunc, sin, cos, pi, nan_to_num

rect = frompyfunc(lambda x, c: float(abs(x)<c), 2, 1)

def freqs(n):
    negs = int((n-1)/2)
    vmax = float(int(n/2))
    return roll(arange(n)-negs, -negs)/vmax

def rfreqs(n):
    return arange(n)/float(n-1)

def hilbert(n,c):
    ks = freqs(n)
    return -1.j*sign(ks)*rect(ks, c)

def ramp(n,c):
    ks = freqs(n)
    return abs(ks)*rect(ks, c)

def shepp_logan(n,c):
    ks = freqs(n)
    return ramp(n,c)*nan_to_num(sin(ks*pi*0.5/c)/(ks*pi*0.5/c), nan=1)

def cosine(n,c):
    ks = freqs(n)
    return abs(ks)*cos(ks*pi*0.5/c)


def rramp(n,c):
    ks = rfreqs((n/2)+1)
    return ks*rect(ks, c)