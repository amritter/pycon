# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)
'''
A collection of utilities in combination with phase reconstruction.
'''
import numpy

def phase_sampling(samples, periods=1., offset=0.):
    '''
    Get a number of phase sampling values in radians over a specified number of
    periods with a given offset.
    
    :param int samples: Number of samples.
    :param float periods: A floating point number specifying the number of
                          periods.
    :param float offset: An offset added to each phase sampling value.
    :returns: A numpy.ndarray containing the phase sampling values.
    :rtype: numpy.ndarray
    '''
    return (2. * periods * numpy.pi / float(samples) * numpy.arange(samples) 
            + offset)