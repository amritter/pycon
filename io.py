# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)
'''
A module collecting utilities to read and write data from and to files.
'''
import numpy
import h5py

class LoadTXT(object):
    '''
    A data loader that uses numpy.loadtxt function to read data from plain text
    files. Objects of class LoadTXT can be used as functions that expect a
    filename to be loaded. The function returns a numpy.array object with the
    content loaded from the file.
    
    :param *args: Additional positional arguments supported by numpy.loadtxt
                  function.
    :param **kwargs: Additional keyword arguments suppoerted by numpy.loadtxt
                     function.
    '''
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    def __call__(self, filename):
        return numpy.loadtxt(filename, *self._args, **self._kwargs)
    
class LoadHDF5(object):
    '''
    A data loader that uses h5py.File objects to read data from HDF5 files.
    Objects of class LoadHDF5 can be used as functions that expect a filename to
    be loaded. The function returns a numpy.array object with the content loaded
    from the file. This class can for example be used to load data from Matlab
    created HDF5 files. 
    
    :param str key: A key that specifies the data object to be extracted from
                    the HDF5 file.
    :param *args: Additional positional arguments supported by h5py.File class
                  except mode which is always 'r' for read only access.
    :param **kwargs: Additional keyword arguments supported by h5py.File class
                     except mode.
    '''
    def __init__(self, key, *args, **kwargs):
        self._key = key
        self.set_args('r', *args, **kwargs)
    def set_args(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
    def __call__(self, filename):
        with h5py.File(filename) as hdf5_file:
            data = numpy.array(hdf5_file[self._key])
        return data
    
def load_ranges(filename, *args, **kwargs):
    '''
    Load data from a range of files containing equal shaped array data.
    
    :param str filename: The common name of the files to be loaded. This may
                         contain any number of "replacement fields" as defined
                         by the Python format string syntax. The field_name of
                         the replacement field can only be positional. For each
                         positional replacement field a range has to be given as
                         an additional argument to load_ranges function. 
    :param *args: Additional arguments containing the ranges for the replacement
                  fields within the given filename. The order in which ranges
                  have to be given is tied to the position of the replacement
                  field.
    :param **kwargs: Additional keyword arguments which are defined below.
    :param callable loader: A callable object that returns the array content
                            which is extracted from the file specified by the
                            filename given to the loader. If not given a LoadTXT
                            object is used to extract data from plain text
                            files.
    :param callable preprocess: A callable object that obtains the data loaded
                                from a single file. Any my do any operations on
                                the given data. The callable has to return the
                                result of the operations. The result should be
                                an numpy.array object. The shape of the returned
                                numpy.array object should be the same for all
                                given input data. Default: None. 
    '''
    loader = kwargs['loader'] if ('loader' in kwargs) else LoadTXT()
    preprocess = kwargs['preprocess'] if ('preprocess' in kwargs) else None
    def _load_ranges(fmt_tuple, *args):
        if len(args):
            return [_load_ranges(fmt_tuple+(value,), *args[1:]) for value in args[0]]
        else:
            data = loader(filename.format(*fmt_tuple) if len(fmt_tuple) else filename)
            return preprocess(data) if preprocess else data
    return numpy.array(_load_ranges(tuple(), *args)) 


        
        
        
        
