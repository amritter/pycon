# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)

from numpy import pi, arange, fft, abs, angle, cos, sin, log, array, transpose
from scipy.optimize import fmin_tnc

def psteprange(steps, periods=1.):
    return 2.*periods * pi / float(steps) * arange(steps)


class fftreco(object):
    def __init__(self, ncoeff=1):
        self._ncoeff = ncoeff
    def __call__(self, array):
        ft = fft.rfft(array)
        size = len(array)
        return [abs(ft[0] / size), angle(ft[self._ncoeff]), abs(2.*ft[self._ncoeff]) / abs(ft[0])]

class fftestimator(fftreco):
    def __init__(self, *args, **kwargs):
        fftreco.__init__(self, *args, **kwargs)
    def __call__(self, array):
        start = fftreco.__call__(self, array)
        bounds = [(start[0] * (1 - .5 * start[2]), start[0] * (1 + .5 * start[2])), (start[1] - 2.*pi, start[1] + 2.*pi), (.1 * start[2], 10.*start[2])]
        return start, bounds
        

class fitreco:
    def __init__(self, objective, ncoeff=1, estimator=fftestimator(), normalizer=None, approx_grad=False, disp=0):       
        self.objective = objective
        self.estimate = estimator
        self.normalize = normalizer        
        if self.normalize == None:
            self.normalize = self.defaultNormalizer
        self.approx_grad = approx_grad
        self.disp = disp                    
    def __call__(self, array):
        fit, bounds = self.estimate(array)        
        return self.normalize(fmin_tnc(self.objective, fit, args=(array,),
                                bounds=bounds, approx_grad=self.approx_grad, disp=self.disp))   
    @staticmethod
    def defaultNormalizer(fit):
        return fit[0]

class Sinoidal:
    def __init__(self, steps):
        self.steps = steps
    def __call__(self, p):
        return p[0] * (1 + p[2] * cos(self.steps + p[1]))
    def prime(self, p):
        c = cos(self.steps + p[1])
        s = sin(self.steps + p[1])
        return transpose([1 + p[2] * c, -p[0] * p[2] * s, p[0] * c])

    
def transmission(obj, ref):
    return obj[0] / ref[0]

def attenuation(obj, ref):
    return 1 - transmission(obj, ref)

def attenuationLength(obj, ref):
    return -log(transmission(obj, ref))

def dphase(obj, ref):
    return obj[1] - ref[1]

def vtransmission(obj, ref):
    return obj[2] / ref[2]

def vattenuation(obj, ref):
    return 1 - vtransmission(obj, ref)

def ndmap(function, *arrays):
    shape = arrays[0].shape
    for it in arrays:
        if it.shape != shape:
            return None
    return _ndmap(function, *arrays)
    
def _ndmap(function, *arrays):
    if arrays[0].ndim > 1:
        return array([_ndmap(function, *it) for it in zip(*arrays)])
    return function(*arrays)
           
        
    
