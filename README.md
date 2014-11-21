pycon
=====

A framework for image reconstruction in X-ray Talbot-Lau interferometry

André Ritter (andre.ritter@fau.de)

Build and install from source
=============================

Prerequisites
-------------

* Python modules: numpy, scipy, matplotlib, h5py
* C++ libraries: boost::python
* Compiler with c++11 support
* If needed create setup.cfg, setting include-dirs and library-dirs

Build
-----

python setup.py build

Install
-------

python setup.py install

Documentation
=============

Extensive documentation is given by docstrings.

Build sphinx documentation
--------------------------

The subdirectory doc contains everything that is needed to build a html or pdf
documentation with sphinx-doc from the content of the docstrings. For this a
working installation of the  sphinx package is needed and the pycon package has
to be importable in python. 

The documentation can be built by typing

make html 

or 

make latexpdf (For this pdflatex has to be installed)

The documentation can then by found in
doc/_build/html/index.html
or
doc/_built/latex/pycon.pdf




