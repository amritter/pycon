# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)

from numpy import arange, abs, sign, roll, frompyfunc

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

def rramp(n,c):
    ks = rfreqs((n/2)+1)
    return ks*rect(ks, c)