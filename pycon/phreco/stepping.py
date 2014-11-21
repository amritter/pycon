# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)

import numpy
from scipy.optimize import fmin_tnc
from scipy.stats import norm

def steppingShift(stepping, stepw, shift):
    rstep = -shift/stepw
    fstep = int(numpy.floor(rstep)) 
    winv = rstep-float(fstep)
    w = 1.-winv
    rolled = numpy.roll(stepping, -fstep)
    rolled = numpy.append(rolled, rolled[0])
    shifted = numpy.zeros_like(stepping)
    for i in range(0,len(rolled)-1):
        shifted[i] = w*rolled[i]+winv*rolled[i+1]
    return shifted

def steppingConv(stepping, shifts, weights, stepw):
    convoluted = numpy.zeros_like(stepping)
    for shift, weight in zip(shifts, weights):
        convoluted += weight*steppingShift(stepping, stepw, shift)
    return convoluted


def skew(x, n, alpha, sigma, offset):
    return n*norm.pdf((x-offset)/sigma)*(1+norm.cdf((alpha*(x-offset))))

def autosampleSkew(n, alpha, sigma, offset, samples=100, width=4.):
    width = width*sigma
    return numpy.arange(offset-.5*width, offset+.5*width, width/samples)
    

def skewSteppingXi(p, stepping, reference, stepw):
    shifts = autosampleSkew(*p)
    weights = skew(shifts, *p)
    return sum((stepping-steppingConv(reference, shifts, weights, stepw))**2)

def convSteppingXi(p, stepping, reference, sampling, stepw):
    return sum((stepping-steppingConv(reference, sampling, numpy.array(p), stepw))**2) 

def fitSkew(stepping, reference, stepw, p0 = numpy.array([1.,0.,.01,0.]), bounds=[(0.,None),(0.,None),(0.,None),(0.,None)]):    
    return fmin_tnc(skewSteppingXi, p0, approx_grad=True, args=(stepping, reference, stepw), bounds=bounds)
    
def fitScattering(stepping, reference, sampling, stepw):
    p0 = numpy.zeros_like(sampling)
    bounds = len(p0)*[(0,None)]  
    return fmin_tnc(convSteppingXi, p0, approx_grad=True, args=(stepping, reference, sampling, stepw), bounds=bounds)
    
def convKernel(a):
    vector = numpy.array(a) 
    kernel = numpy.zeros((len(a), len(a)))
    for i in range(len(a)):
        kernel[i] = numpy.roll(vector, i)
    return numpy.transpose(kernel)
         
def convL(x, y, M):
    s = numpy.dot(M, x)
    return -sum(s*numpy.log(y)-y), numpy.dot(numpy.transpose(M), (1-s/y))