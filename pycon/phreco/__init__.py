# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)
'''
The phreco module collects utilities to reconstruct parameters from phase
sampling curves.

The reconstruction is done on numpy.ndarray objects with any number of
dimensions with functions in this module starting by the prefix ``reco_``. In
general it is assumed that algorithms return a mean, a phase and the
visibility aligned along the zeroth axis of the returned numpy.ndarray object.
Thus index 0 gives the mean, index 1 gives the phase and index 2 gives the 
visibility. 
'''
import numpy

def mean(arr):
    '''
    Get the mean values.
    
    :param numpy.ndarray arr: An array containing all parameters aligned along
                              axis 0.
    :returns: ``arr[0]``
    '''
    return arr[0]

def phase(arr):
    '''
    Get the phase values.
    
    :param numpy.ndarray arr: An array containing all parameters aligned along
                              axis 0.
    :returns: ``arr[1]``
    '''
    return arr[1]

def visibility(arr):
    '''
    Get the visibility values.
    
    :param numpy.ndarray arr: An array containing all parameters aligned along
                              axis 0.
    :returns: ``arr[2]``                        
    '''
    return arr[2]

def transmission(arr, ref):
    '''
    Get the transmission as defined by the quotient of arr and ref.
    
    :param numpy.ndarray arr: An array containing only a single parameter type.
    :param numpy.ndarray ref: An array containing only a single parameter type
                             with reference values.
    :returns: ``arr / ref``
    '''
    return arr / ref

def absorption(arr, ref):
    '''
    Get the absorption which is defined by one minus the transmission of arr
    and ref.
    
    :param numpy.ndarray arr: An array containing only a single parameter type.
    :param numpy.ndarray ref: An array containing only a single parameter type
                             with reference values.
    :returns: ``1. - arr / ref``
    '''
    return 1. - transmission(arr, ref)

def attenuation(arr, ref):
    '''
    Get the attenuation which is defined by the negative logarithm of the
    transmission of arr and ref.
    
    :param numpy.ndarray arr: An array containing only a single parameter type.
    :param numpy.ndarray ref: An array containing only a single parameter type
                             with reference values.
    :returns: ``-log( arr / ref )``
    '''
    return -numpy.log(transmission(arr, ref))

def difference(arr, ref):
    '''
    Get the difference of arr and ref.
    
    :param numpy.ndarray arr: An array containing only a single parameter type.
    :param numpy.ndarray ref: An array containing only a single parameter type
                             with reference values.
    :returns: ``arr - ref``
    '''
    return arr - ref

def deref(arr, ref, deref_mean=None, deref_phase=None,
          deref_visibility=None):
    '''
    Get the phase sampling parameters corrected by reference values.
    
    :param numpy.ndarray arr: The array with parameters to be corrected. The
                              parameters have to be aligned along the zeroth
                              axis.
    :param numpy.ndarray ref: The array with reference parameters for
                              correction. The parameters have to be aligned
                              along the zeroth axis.
    :param callable deref_mean: A function taking an numpy.ndarray of mean
                                values and a second numpy.ndarray of reference
                                values. The function returns the corrected
                                values. If None pycon.phreco.transmission is
                                taken as default.
    :param callable deref_phase: A function taking an numpy.ndarray of phase
                                 values and a second numpy.ndarray of reference
                                 values. The function returns the corrected
                                 values. If None pycon.phreco.difference is
                                 taken as default.
    :param callable deref_visibility: A function taking an numpy.ndarray of
                                      visibility values and a second
                                      numpy.ndarray of reference values. The
                                      function returns the corrected values. If
                                      None pycon.phreco.transmission is taken as
                                      default.
    '''
    deref_mean = deref_mean if deref_mean else transmission
    deref_phase = deref_phase if deref_phase else difference
    deref_visibility = deref_visibility if deref_visibility else transmission
    return numpy.array([deref_mean(mean(arr), mean(ref)),
                        deref_phase(phase(arr), phase(ref)),
                        deref_visibility(visibility(arr), visibility(ref))])

def reco_fft(arr, order=1, axis=0):
    '''
    Reconstruct mean, phase and visibility from phase sampling data.
    
    :param numpy.ndarray arr: input array containing phase sampling data aligned
                              along the specified axis.
    :param int order: The order of the harmonics that contains phase and
                      visibility.
    :param int axis: The axis along which the phase sampling is aligned.
    :returns: A numpy.ndarray with the reconstructed mean, phase and visibility.
              The array has the same number of dimensions. The reconstructed
              parameters are aligned along the zeroth axis of the returned
              array.
    :rtpye: numpy.ndarray
    '''   
    ft = numpy.rollaxis(numpy.fft.rfft(arr, axis=axis), axis)
    norm = 1. / arr.shape[axis]
    return numpy.array([numpy.abs(ft[0] * norm),
                        numpy.angle(ft[order]),
                        abs(2.*ft[order] / ft[0])])
