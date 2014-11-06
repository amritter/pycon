# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)

from os import listdir
from numpy import array, loadtxt, transpose
import fnmatch

def count(predicate, sequence):
    return reduce(lambda x, y: int(x)+int(y), map(predicate, sequence))

def getTLISteps(prefix, files):    
    psteps = count(lambda x: fnmatch.fnmatch(x, prefix+'0000*'), files) 
    files = count(lambda x: fnmatch.fnmatch(x, prefix+'*'), files)
    if psteps == 0 or files == 0:
        return 0
    return files/psteps    

def loadTLIProjections(prefix, steps=None):
    dirname, sep, fnprefix = prefix.rpartition('/')
    if sep == '' and dirname == '':
        dirname = './'
        if fnprefix == '' :
            return None
    dirfiles = listdir(dirname)     
    if steps == None:
        steps = range(getTLISteps(fnprefix, dirfiles))    
    data = []              
    for step in steps:
        stepprefix = fnprefix+str(step).zfill(4)+'*'
        stepfiles = [dirname + sep + fn for fn in fnmatch.filter(dirfiles, stepprefix)]
        projection = []
        for stepfile in stepfiles:        
            projection.append(loadtxt(stepfile, comments='#'))    
        data.append(transpose(projection))
    return array(data)
    
        