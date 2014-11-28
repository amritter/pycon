# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)

import numpy
import numpy.fft as fft
import scipy.interpolate as sip
import kernel

def xi(x, y, theta):
    return x*numpy.cos(theta)-y*numpy.sin(theta)

def pad(arr, pads=0, value=0., axis=None):
    pad_arr = value*numpy.ones(int(pads))
    def _pad(subarr):
        return numpy.concatenate((pad_arr, subarr, pad_arr))
    if axis == None:
        axis = arr.ndim-1    
    return numpy.apply_along_axis(_pad, axis, arr)
    
def unpad(arr, pads=0, axis=None):
    arr = numpy.array(arr)
    if axis == None:
        axis = arr.ndim-1
    if axis < 0:
        axis = arr.ndim+axis
    return numpy.rollaxis(numpy.rollaxis(arr, axis)[pads:-pads], 0, axis+1)

def ramp(c=1.):
    return lambda n: kernel.ramp(n, c)

def hilbert(c=1.):
    return lambda n: kernel.hilbert(n, c)

def filter_projections(array, kernel=ramp(1.), axis=None):
    if axis == None:
        axis = array.ndim-1
    return numpy.real(fft.ifft(fft.fft(array, axis=axis)
                               *kernel(array.shape[axis]), axis=axis))

def backproject_projection(projection, theta, xis, xs, ys,
                                    kind='linear'):
    return sip.interp1d(xis, projection, fill_value=0, copy=False, kind=kind,
                      bounds_error=False, assume_sorted=True)(xi(xs, ys, theta))

def backproject(sinogram, thetas, xis, xs, ys, interpolation='linear'):
    crosssection = numpy.zeros_like(xs)
    for projection, theta in zip(sinogram, thetas):
        crosssection += backproject_projection(projection, theta, xis,
                                                     xs, ys, kind=interpolation)
    return crosssection/len(thetas)
        