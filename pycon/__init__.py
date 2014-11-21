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

def show_doc_html(refresh=False):
    '''
    Show the documentation. This needs a webbrowser to be installed.
    Additionally the function tries to refresh the documentation if wanted.
    
    :param bool refresh: If set to True the function tries to refresh the
                         documentation before showing it.
    ''' 
    import webbrowser
    import os
    import os.path
    import subprocess
    package_root = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    html_index = os.path.join(package_root, '../doc/_build/html/index.html')
    # Try to update documentation if needed.
    if refresh:
        cwd_old = os.getcwd()
        os.chdir(os.path.join(package_root, '../doc'))
        if subprocess.call(['make', 'html']) == 0:
            print 'Html documentation refreshed.'
        else:
            print ('Could not refresh html documentation. Maybe sphinx-doc is '
                   'not installed on this system.')
        os.chdir(cwd_old)
    # Try showing documentation.
    if os.path.exists(html_index):
        print 'Showing sphinx html documentation at: '+html_index
        webbrowser.open('file://'+html_index)
    else:
        print 'Could not find html documentation index at: '+html_index
    