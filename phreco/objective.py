# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)

from numpy import sum, dot, log 

def logLPoisson(model, p, y):
    f = model(p)
    return sum(f-y*log(f))

def logLPoissonGrad(model, p, y):    
    return dot(y/model(p)-1., model.prime(p))

def logLPoissonAndGrad(model, p, y):
    f = model(p)
    return sum(f-y*log(f)), dot(1.-y/f, model.prime(p))

def xiSquared(model, p, y):
    return sum((y-model(p))**2)

def xiSquaredAndGrad(model, p, y):
    res = y-model(p)
    return sum(res**2), -2.*dot(res, model.prime(p)) 

def objective(objective, model):
    return lambda *p: objective(model, *p)