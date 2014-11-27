# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)

import numpy
import _fp
import matplotlib.pyplot as pl
import scipy.optimize as opt
import time

n0_default = 12
n1_default = 12
pitch0_default = .1
pitch1_default = .1
angles_default = 101


thetas = numpy.arange(0., 2.*numpy.pi, 2.*numpy.pi/angles_default)
xis = numpy.arange(-.45+.025, .5, .1)

def get_projectors(thetas, xis, xipitch, n0=n0_default, n1=n1_default, pitch0=pitch0_default, pitch1=pitch1_default, offset0=0., offset1=0.):
    width0 = n0 * pitch0
    width1 = n1 * pitch1
    p = _fp.projector_siddon2d(thetas, xis, n0, n1, width0, width1, offset0, offset1)
    p_diff = _fp.projector_siddon2d(thetas, xis, n0, n1, width0, width1, offset0, offset1, xis_diff=xipitch*numpy.ones_like(xis)) 
    p_re = p.transposed()
    p_diff_re = p_diff.transposed()
    return ((p, p_re), (p_diff, p_diff_re), (thetas, xis), (n0, n1))

def phi0_random_uniform(thetas, xis, b_lo=0., b_hi=2.*numpy.pi):
    return numpy.random.uniform(b_lo, b_hi, (len(thetas), len(xis)))

def empty_volume(n0=n0_default, n1=n1_default):
    volume = numpy.zeros((3, n1, n0))
    return volume

def reshape_volume(vol, n0, n1):
    return numpy.reshape(vol, (3, n1, n0))
    
def phantom_boxes(n0=n0_default, n1=n1_default, n0_border=2, n1_border=2, vmax=(1., 1., 1.)):
    volume = empty_volume()
    volume[0,1:11,1:6] = 1.*vmax[0]
    volume[0,1:11,6:11] = .5*vmax[0]
    volume[1,1:6,1:11] = 1.*vmax[1]
    volume[1,6:11,1:11] = .5*vmax[1]
    volume[2,2:5,2:5] = 1.*vmax[2]
    volume[2,7:10,7:10] = .5*vmax[2]
    return volume
    
def phantom_random(n0=n0_default, n1=n1_default, n0_border=2, n1_border=2, bounds=((0.,1.),(0.,1.),(0.,1.))):
    volume = empty_volume(n0, n1)
    for i in range(3):
        volume[i,n1_border:-n1_border,n0_border:-n0_border] = numpy.random.uniform(bounds[i][0], bounds[i][1], (n1-2*n1_border, n0-2*n0_border))
    return volume
    
def forward_project(volume, projectors):
    volume = reshape_volume(volume, *projectors[3])
    return numpy.array([numpy.exp(-projectors[0][0].project(volume[0])),
                        .5*projectors[1][0].project(volume[1]),
                        numpy.exp(-projectors[0][0].project(volume[2]))])
                
def volume_project(projection, projectors):
    return numpy.array([projectors[0][1].project(projection[0]),
                        .5*projectors[1][1].project(projection[1]),
                        projectors[0][1].project(projection[2])])
    
def get_phis(phi, phi0, psteps):
    return numpy.repeat(phi[numpy.newaxis], len(psteps), axis=0)+psteps[:,numpy.newaxis,numpy.newaxis]+phi0
    
def stepping(N0, V0, projection, phi0, psteps):
    phis = get_phis(projection[1], phi0, psteps)
    N = N0*projection[0]*(1.+V0*projection[2]*numpy.cos(phis))
    return N
    
def loglikelihood(N, N_exp):
    return numpy.sum(N_exp-N*numpy.log(N_exp)+N*numpy.log(N)-N)
    
def ll(volume, N, N0, V0, phi0, projectors, psteps):
    N_exp = stepping(N0, V0, forward_project(volume, projectors), phi0, psteps)
    return loglikelihood(N, N_exp)
    
def llgradient(volume, N, N0, V0, phi0, projectors, psteps):
    projection = forward_project(volume, projectors)
    phis = get_phis(projection[1], phi0, psteps)
    C = numpy.cos(phis)
    T = N0*projection[0]
    TD = T*V0*projection[2]
    N_exp = T+TD*C
    N_ratio = numpy.nan_to_num(1.-N/N_exp)
    return -volume_project([
        numpy.sum(N_exp-N, axis=0),
        TD*numpy.sum(N_ratio*numpy.sin(phis), axis=0),
        TD*numpy.sum(N_ratio*C, axis=0)],
        projectors)
        
def ll3eps(eps, volume, grad, N, N0, V0, phi0, projectors, psteps):
    return ll(volume+numpy.abs(numpy.array(eps))[:,numpy.newaxis,numpy.newaxis], N, N0, V0, phi0, projectors, psteps)
        
fmin_methods = dict(
    cg=opt.fmin_cg,
    bfgs=opt.fmin_bfgs
    )
        
def reconstruct(N, N0, V0, phi0, projectors, psteps, method='bfgs',
                gtol=1e-5, full_output=False, retall=False):
    start = empty_volume(*projectors[3])
    norm = 1./abs(ll(start, N, N0, V0, phi0, projectors, psteps))
    ret = fmin_methods[method](
                    lambda *args: norm*ll(*args),
                    #ll,
                    start,
                    lambda *args: norm*llgradient(*args).flatten(),
                    #lambda *args: llgradient(*args).flatten(),
                    args=(N, N0, V0, phi0, projectors, psteps),
                    gtol=gtol,
                    full_output=full_output,
                    retall=retall)
    if full_output or retall:
        return reshape_volume(ret[0], *projectors[3]), ret[1:]
    else:
        return reshape_volume(ret, *projectors[3])
        
def iterate(volume, N, N0, V0, phi0, projectors, psteps, method='cg'):
    norm = 1./abs(ll(volume, N, N0, V0, phi0, projectors, psteps))
    grad = norm*llgradient(volume, N, N0, V0, phi0, projectors, psteps)
    eps = fmin_methods[method](
            lambda *args: norm*ll3eps(*args)-1.,
            [0,0,0],
            args=(volume, grad, N, N0, V0, phi0, projectors, psteps)
        )
    return volume+numpy.abs(numpy.array(eps))[:,numpy.newaxis,numpy.newaxis]*grad, eps
    
    
def show_volume(volume):
    pl.figure()
    pl.subplot(131)
    pl.imshow(volume[0], interpolation='nearest', cmap=pl.cm.gray)
    pl.subplot(132)
    pl.imshow(volume[1], interpolation='nearest', cmap=pl.cm.gray)
    pl.subplot(133)
    pl.imshow(volume[2], interpolation='nearest', cmap=pl.cm.gray)
    
def write_movie(fprefix, data, vmin, vmax, scale=(1., 1., 1.), fps=15, n0=n0_default, n1=n1_default):
    def frame_data(frame_number):
        return numpy.reshape([scale[0]*data[frame_number,0,:,:].T,
                scale[1]*data[frame_number,1,:,:].T,
                scale[2]*data[frame_number,2,:,:].T], (3*n1,n0)).T
    for i in range(len(data)):
        pl.imsave(fprefix+'_frame_{:0=3}.png'.format(i), frame_data(i), vmin=vmin, vmax=vmax, cmap=pl.cm.gray, format='png')
    
def timeit(func, *args, **kwargs):
    t_s = time.clock()
    ret = func(*args, **kwargs)
    t_e = time.clock()
    print 'Time needed: '+str(t_e-t_s)+' s'
    return ret