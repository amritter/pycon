# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)
'''
This is the pycon Python package. A framework for image reconstruction in X-ray
Talbot-Lau interferometry.

Show the documentation with pycon.show_doc_html().
'''

from io import load_ranges

def show_doc_html():
    '''
    Show the documentation. This needs a webbrowser to be installed and the html
    documentation in the doc subdirectory of the pycon package to be built.
    ''' 
    import webbrowser
    import os.path
    package_root = os.path.dirname(os.path.abspath(__file__))
    html_index = os.path.join(package_root, 'doc/_build/html/index.html')
    print 'Showing sphinx html documentation at: '+html_index
    webbrowser.open('file://'+html_index)