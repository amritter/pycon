# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)

from distutils.core import setup, Extension

fp = Extension('fp',
                    libraries = ['boost_python', 'm'],
                    extra_compile_args = ['-std=c++11'],
                    sources = ['fp.cc'])

setup (name = 'pycon',
       version = '1.0',
       description = '',
       ext_modules = [fp])