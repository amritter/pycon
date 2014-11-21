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
    return x*numpy.cos(theta)+y*numpy.sin(theta)

def zero_pad(array, pads=0., value=0., axis=None):
    array = numpy.array(array)
    if axis == None:
        axis = array.ndim-1
    return numpy.insert(array, pads*[array.shape[axis]]+pads*[0], value,
                        axis=axis)

def ramp(c):
    return lambda n: kernel.ramp(n, c)

def hilbert(c):
    return lambda n: kernel.hilbert(n, c)

def filter_parallel_projections(array, kernel=ramp(1.), axis=None):
    if axis == None:
        axis = array.ndim-1
    print axis
    print array.shape
    print array.shape[axis]
    return numpy.real(fft.ifft(fft.fft(array, axis=axis)
                               *kernel(array.shape[axis]), axis=axis))

def backproject_parallel_projection(projection, theta, xis, xs, ys,
                                    kind='linear'):
    return sip.interp1d(xis, projection, fill_value=0, copy=False, kind=kind,
                      bounds_error=False, assume_sorted=True)(xi(xs, ys, theta))

def backproject_parallel(sinogram, thetas, xis, xs, ys, interpolation='linear'):
    crosssection = None
    for projection, theta in zip(sinogram, thetas):
        backprojection = backproject_parallel_projection(projection, theta, xis,
                                                     xs, ys, kind=interpolation)
        if crosssection == None:
            crosssection = backprojection
        else:
            crosssection += backprojection
    return crosssection/len(thetas)
        