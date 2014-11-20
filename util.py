# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)
'''
A module collecting some general utilities to process data.
'''

class Chain(list):
    '''
    A processing chain. Objects of this class are callable on numpy.ndarray
    objects and return a numpy.ndarray object. The returned object is the result
    of a consecutive application of a list of callables, where each returned 
    numpy.ndarray object is the input for the next callable.
    
    :param args: The list of callables given as additional arguments.
                 Processing is done in the order as specified by the argument
                 order.
    
    The follwing example is for constructing a processing chain, that first
    applies a detector correction then crops the image data and then rebins the
    image. The input data assumes an ndarray with ndim = 3, where axis 1 and 2
    are for the detector matrix and axis 0 is for some different parameter
    variation.
    
    Example::
    
        chain = pycon.util.Chain(
                    # Assume given gain and offset.
                    pycon.util.DetectorCorrection(gain, offset),
                    # Crop on axis 1 and 2, leave axis 0.
                    pycon.util.Slice()[:,10:-10, 10:-10],
                    # Rebin 2x2 elements for axis 1 and 2.
                    pycon.util.Rebin((2, 1), (2, 2)))
    '''
    def __init__(self, *args):
        list.__init__(self, *args)
    def __call__(self, arr):
        for processor in self:
            arr = processor(arr)
        return arr

class Slice(object):
    '''
    A slicing processor. Objects of this class are callable on numpy.ndarray
    objects and return a numpy.ndarray object that is sliced according to the
    numpy slicing syntax. Slicing is set by using the slicing operator of this
    class.
    
    Example input array::
    
        >>> numpy.arange(16.).reshape((4,4))
        array([[  0.,   1.,   2.,   3.],
               [  4.,   5.,   6.,   7.],
               [  8.,   9.,  10.,  11.],
               [ 12.,  13.,  14.,  15.]])
               
    Create a slicing processor that creates slices where a boundary region of
    one element on each axis is removed::
    
        >>> s = pycon.util.Slice()[1:-1,1:-1]
        >>> s(numpy.arange(16.).reshape((4,4)))
        array([[  5.,   6.],
               [  9.,  10.]])
    '''
    def __init__(self):
        self._slicing = None
    def __getitem__(self, slicing):
        self._slicing = slicing
        return self
    def __call__(self, arr):
        return arr[self._slicing]

def rebin(arr, bins, axis=-1):
    '''
    Rebin number of bins along a given axis.
    
    :param numpy.ndarray arr: A ndarray object.
    :param int bins: Number of bins that are averaged to one bin.
    :param int axis: The axis along which the rebinning is done. Default: -1
                     (last axis).
    :returns: The rebinned ndarray object.
    :rtype: numpy.ndarray
    
    Example input array::
    
        >>> numpy.arange(16.).reshape((4,4))
        array([[  0.,   1.,   2.,   3.],
               [  4.,   5.,   6.,   7.],
               [  8.,   9.,  10.,  11.],
               [ 12.,  13.,  14.,  15.]])
               
    Rebinning along default (last) axis::
    
        >>> pycon.util.rebin(numpy.arange(16.).reshape((4,4)), 2)
        array([[  0.5,   2.5],
               [  4.5,   6.5],
               [  8.5,  10.5],
               [ 12.5,  14.5]])
    
    Rebinning along first axis::
             
        >>> pycon.util.rebin(numpy.arange(16.).reshape((4,4)), 2, 0)
        array([[  2.,   3.,   4.,   5.],
               [ 10.,  11.,  12.,  13.]])  
    '''
    axis = arr.ndim + axis if axis < 0 else axis    
    shape = arr.shape[:axis]+(arr.shape[axis]/bins, bins)+arr.shape[axis+1:]
    return arr.reshape(shape).mean(axis+1)
    
def rebin_multi(arr, *args):
    '''
    Rebin along multiple given axes. Equal to repeatedly using pycon.util.rebin.
    
    :param numpy.ndarray arr: A ndarray object.
    :param args: Each additionaly argument is a tuple of two int values. The
                 first tuple element specifies the number of bins to average.
                 The second tuple element specifies the axis along which to
                 rebin.
    :returns: The rebinned ndarray object.
    :rtype: numpy.ndarray
    
    Example input array::
    
        >>> numpy.arange(16.).reshape((4,4))
        array([[  0.,   1.,   2.,   3.],
               [  4.,   5.,   6.,   7.],
               [  8.,   9.,  10.,  11.],
               [ 12.,  13.,  14.,  15.]])
    
    Rebinning 2x2 bins:
    
        >>> pycon.util.rebin_multi(numpy.arange(16.).reshape((4,4)),
                                   (2, 0), (2, 1))
        array([[  2.5,   4.5],
               [ 10.5,  12.5]])
    '''
    for arg in args:
        arr = rebin(arr, *arg)
    return arr
    
class Rebin(object):
    '''
    A rebinning processor. Objects of this class are callable on numpy.ndarray
    objects and return a numpy.ndarray object rebinned with
    pycon.util.rebin_multi function.
    
    :param args: The same additional arguments that would be used for directly
                 calling pycon.uti.rebing_multi. Effectively setting the
                 rebinning parameters.
    '''
    def __init__(self, *args):
        self._args = args
    def __call__(self, arr):
        return rebin_multi(arr, *self._args)
    
class DetectorCorrection(object):
    '''
    A simple detector correction processor. Objects of this class are callable
    on numpy.ndarray objects and return a numpy.ndarray object that is corrected
    for detector gain, detector offset or both. To use this correction either
    gain, offset or both have to be set.
    
    :param numpy.ndarray gain: A numpy.ndarray object that describes the
                               gainfactor of each detector pixel. The gain can
                               be obtained by a measurement of the detector
                               readout value for iradiated measurements
                               corrected for an offset.
    :param numpy.ndarray offset: A numpy.ndarray object that describes the 
                                 offset of each detector pixel as defined by
                                 the mean dark readout value.
    '''
    def __init__(self, gain=None, offset=None):
        self._gainc = 1./gain if gain else None
        self._offset = offset
        if self._gainc and self._offset:
            if self._gainc.shape != self._offset.shape:
                raise ValueError('Shape of gain and offset array do not match')
            self.__call__ = self._corr_gain_and_offset
        elif self._gainc:
            self.__call__ = self._corr_gain
        elif self._offset:
            self.__call__ = self._corr_offset
        else:
            ValueError('Either gain or offset or both have to be set.')
    def _corr_gain(self, arr):
        return self._gainc*arr
    def _corr_offset(self, arr):
        return arr-self._offset
    def _corr_gain_and_offset(self, arr):
        return self._gainc*(arr-self._offset)