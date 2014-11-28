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
import fbp
import numpy
import pycon.util

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
    if not thetas:  # Assume projections over 2 pi.
        thetas = numpy.arange(0., 2.*numpy.pi, 2.*numpy.pi / sino.shape[thetas_axis])
    if not xis:
        xis = numpy.arange(sino.shape[xis_axis]) - .5 * (sino.shape[xis_axis] - 1)+xi_offs
    if not xy:
        xi_max = numpy.max(numpy.abs(xis))
        xi_diff_max = numpy.max(numpy.diff(xis))
        n_coords = int(numpy.round(2.*xi_max / xi_diff_max))
        coords = numpy.arange(n_coords) - .5 * (n_coords - 1)
        xy = numpy.meshgrid(coords, coords)
    if not pads:
        pads = int(.5 * sino.shape[xis_axis])
    if not kernel:
        kernel = fbp.ramp()
    sino_filtered = fbp.unpad(fbp.filter_projections(fbp.pad(sino, pads, pad_value, xis_axis), kernel, xis_axis), pads, xis_axis)
    x, y = xy
    return pycon.util.apply_along_axes(fbp.backproject, (thetas_axis, xis_axis),
                                       sino_filtered, thetas, xis, x, y, interpolation)

