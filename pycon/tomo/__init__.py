# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)
'''

2D-tomo coordinates

.. math::

    x = \\lambda\\cdot\\sin(\\theta) + \\xi\\cdot\\cos(\\theta) 
    
    y = \\lambda\\cdot\\cos(\\theta) - \\xi\\cdot\sin(\\theta)
'''
from . import fbp
import numpy
import scipy.interpolate
import skimage.restoration
import pycon.util
import pycon.phreco

def reco_fbp(sino, thetas=None, xis=None, xi_offs=0., xy=None, kernel=None, pads=None, 
             pad_value=0., interpolation='linear', thetas_axis=-2, xis_axis=-1):
    '''
    Reconstruct a cross section from tomographic parallel projection data using
    the filtered back projection method.
    
    :param numpy.ndarray sino: An array containing the sinogram data.
    :param numpy.ndarray thetas: An array specifing the angle in radians of each
                                 parallel projection. If set to None, angles are
                                 distributed between 0 and 2 pi according to the
                                 number of projections in sino.
    :param numpy.ndarray xis: An array specyifing the xi values of the pixels in
                              each parallel projection. If set to None, a pitch
                              of one is assumed and a number of xis specified by
                              the shape of sino is distributed around zero.
    :param float xi_offs: If xis are set to None this value allows to shift the
                          computed xis by the given offset. This allows to
                          correct for a not centered rotation axis.
    :param tuple xy: A tuple of two equal shaped 2d numpy.ndarray objects. The
                     shape reflects the shape of the reconstruced x-y-volume.
                     The first array gives the x coordinates the second array
                     gives the y coordinates. If set to None, a volume is
                     assumed which is defined by ``max(abs(xis))`` for its outer
                     boundaries and ``max(diff(xis))'' for its voxel size.
    :param kernel: A kernel object accepted by pycon.tomo.filter_projection
                   method.
    :param int pads: The number of pad values to be inserted at the front and 
                     back of the xi-axis. If set to None, one half of the length
                     of the xi-axis is assumed.
    :param float pad_value: The pad value to be inserted. Default: 0.
    :param interpolation: A value accepted by scipy.interpolate.interp1d as 
                          kind argument.
    :param thetas_axis: Specifies the position of the theta-axis in sino.
    :param xis_axis: Specifies the position of the xi-axis in sino.
    '''
    if thetas is None:  # Assume projections over 2 pi.
        thetas = numpy.arange(0., 2.*numpy.pi, 2.*numpy.pi / sino.shape[thetas_axis])
    if xis is None:
        xis = numpy.arange(sino.shape[xis_axis]) - .5 * (sino.shape[xis_axis] - 1)+xi_offs
    if xy is None:
        xi_max = numpy.max(numpy.abs(xis))
        xi_diff_max = numpy.max(numpy.diff(xis))
        n_coords = int(numpy.round(2.*xi_max / xi_diff_max))
        coords = numpy.arange(n_coords) - .5 * (n_coords - 1)
        xy = numpy.meshgrid(coords, coords)
    if pads is None:
        pads = int(.5 * sino.shape[xis_axis])
    if kernel is None:
        kernel = fbp.ramp()
    sino_filtered = fbp.unpad(fbp.filter_projections(fbp.pad(sino, pads, pad_value, xis_axis), kernel, xis_axis), pads, xis_axis)
    x, y = xy
    return pycon.util.apply_along_axes(fbp.backproject, (thetas_axis, xis_axis),
                                       sino_filtered, thetas, xis, x, y, interpolation)
    
def sino_interp(arr, thetas, thetas_new, theta_axis, kind=3):
    arr[1] = skimage.restoration.unwrap_phase(arr[1])
    interp = numpy.apply_along_axis(
                lambda value: scipy.interpolate.interp1d(thetas, value,
                                                         kind=kind,
                                                         bounds_error=False)
                                                         (thetas_new),
                theta_axis, arr)
    interp[1] = pycon.phreco.wrap(interp[1])
    return interp

def sino_polyinterp(arr, thetas, thetas_new, theta_axis, deg=3):
    def _polyinterp(a):
        return numpy.polyval(numpy.polyfit(thetas, a, deg), thetas_new)
    arr[1] = skimage.restoration.unwrap_phase(arr[1])
    interp = numpy.apply_along_axis(_polyinterp, theta_axis, arr)
    interp[1] = pycon.phreco.wrap(interp[1])
    return interp

def sino_dpcenter(arr, axis=-1):
    arr[1] = arr[1] - numpy.expand_dims(arr[1].mean(axis), axis)
    return arr

def sino_dppolycorrect(sino, axis=-1, deg=1):
    x = numpy.arange(sino.shape[axis])
    def _polycorrect(arr):
        return arr-numpy.polyval(numpy.polyfit(x, arr, deg), x)
    return numpy.apply_along_axis(_polycorrect, axis, sino)

def estimate_cor(sino, axis=-1, as_offset=True):
    '''
    Estimate center of rotation for a given sinogram. The projection angles have
    to be distributed equidistantly over interval lengths of integer multiples
    of 2 pi.
    
    :param numpy.ndarray sino: A sinogram.
    :param int axis: The xi axis.
    :param bool as_offset: If True the center of rotation is given as a pixel
                           offset to the center of the xi axis. If False the
                           center of rotation is given as the absolute pixel
                           position on the xi axis.
    :returns: The center of rotation either as absolute value or as offset, both
              in terms of a pixel position on the xi axis.
    :rtype: float
    '''
    axes = tuple(numpy.delete(numpy.arange(sino.ndim), axis, 0))
    m = sino.sum(axes)
    c = numpy.sum(m*numpy.arange(len(m)))/m.sum()
    return c-.5*(len(m)-1) if as_offset else c
