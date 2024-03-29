# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)

from distutils.core import setup, Extension
import numpy

pycon_tomo__fp = Extension('pycon.tomo._fp',
                    include_dirs = [numpy.get_include()],
                    libraries = ['m'],
                    extra_compile_args = ['-std=c++11'],
                    sources = ['pycon/tomo/_fp.cc',
                               'pycon/tomo/siddon.cc'])

setup (name = 'pycon',
       version = '0.0',
       description = 'pycon - A python framework for image reconstruction in '
                     'X-ray Talbot-Lau interferometry.',
       author = 'André Ritter',
       author_email = 'andre.ritter@fau.de',
       packages = ['pycon', 
                   'pycon.phreco',
                   'pycon.tomo'],
       ext_modules = [pycon_tomo__fp])