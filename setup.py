# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)

from distutils.core import setup, Extension

pycon_tomo_fp = Extension('pycon.tomo.fp',
                    libraries = ['boost_python', 'm'],
                    extra_compile_args = ['-std=c++11'],
                    sources = ['pycon/tomo/fp.cc'])

setup (name = 'pycon',
       version = '1.0',
       description = 'pycon - A python framework for image reconstruction in '
                     'X-ray Talbot-Lau interferometry.',
       packages = ['pycon', 
                   'pycon.phreco',
                   'pycon.tomo'],
       ext_modules = [pycon_tomo_fp])