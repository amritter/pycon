# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)

from numpy import zeros, ones, cos, sin, real, concatenate, arange, meshgrid
from numpy.fft import fft, ifft, rfft, irfft
from scipy.interpolate import interp1d
import kernel

def convolute(projection, kernel, ft=fft, ift=ifft):
    fp = ft(projection) 
    if len(kernel) == len(fp):      
        return ift(fp*kernel)
    return None

def rconvolute(projection, kernel):
    return convolute(projection, kernel, ft=rfft, ift=irfft)

def projection_filter(projection, kernel, pads=0, padvalue=0.):
    projection = projection_pad(projection, pads, padvalue)
    if len(projection) == len(kernel):
        return convolute(projection, kernel, ft=fft, ift=ifft)
    elif int(len(projection)/2)+1 == len(kernel):
        return real(convolute(projection, kernel, ft=rfft, ift=irfft))
    return projection

def xi(x, y, theta):
    return x*cos(theta)+y*sin(theta)

def pad(sinogram, pads, value=0.):
    padding = value*ones(int(pads))
    return [concatenate((padding,s,padding)) for s in sinogram]

def ramp(c):
    return lambda n: kernel.rramp(n, c)

def hilbert(c):
    return lambda n: kernel.hilbert(n, c)


def fbp(sinogram, thetas, xis, xs, ys, interpolation='linear', kfun=ramp(1.)):
    tomo = zeros((len(ys), len(xs)))
    kernel = zeros(0)
    if kfun != None:
        kernel = kfun(len(xis))
    f = [interp1d(xis, projection_filter(s, kernel), kind=interpolation, fill_value=0, bounds_error=False) for s in sinogram]
    for ny, y in enumerate(ys):
        for nx, x in enumerate(xs):
            for ntheta, theta in enumerate(thetas):
                tomo[ny][nx] += f[ntheta](xi(x, y, theta))
    return tomo

def filter_kernel(n, kfun):
    return kfun(n)

def projection_pad(projection, pads, value=0.):
    padding = value*ones(int(pads))
    return concatenate((padding,projection,padding))

def projection_unpad(projection, pads):
    return projection[pads:-pads]

def projection_interpolator1d(xis, projection, kind='linear', fill_value=0):
    return interp1d(xis, projection, kind=kind, fill_value=fill_value, bounds_error=False)

def sinogram_interpolator1d(xis, sinogram, kind='linear', fill_value=0.):
    f = lambda projection: projection_interpolator1d(xis, projection, kind, fill_value)
    return map(f, sinogram)

def backproject_pixel(coords, thetas, xis, sinogram, interpolation='linear'):
    if coords == None:
        return 0
    fs = sinogram_interpolator1d(xis, sinogram, interpolation)
    x, y = coords
    return sum(map(lambda f, theta: f(xi(x, y, theta)), fs, thetas))
    
def coords_grid(nx, ny, xwidth=1., ywidth=1., xcentre=0., ycentre=0.):
    x = (arange(nx)-.5*(nx-1))*xwidth-xcentre
    y = (arange(ny)-.5*(ny-1))*ywidth-ycentre
    X, Y = meshgrid(x, y)
    return zip(X.flat, Y.flat)
    
    