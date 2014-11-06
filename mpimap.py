# -*- coding: utf-8 -*-
# This file is part of pycon: A framework for image reconstruction in X-ray
# Talbot-Lau interferometry.
#
# Copyright (C)
# 2012-2014 André Ritter (andre.ritter@fau.de)

from mpi4py import MPI

def identity(x):
	return x

class MapperParent:
	def __init__(self, maxprocs=4):		
		import sys
		self.childrunner = None
		self.create_childrunner()
		self.comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=[self.childrunner],
                           maxprocs=maxprocs)	
	def close(self):
		from os import remove
		self.comm.bcast(None, root=MPI.ROOT)
		remove(self.childrunner)
		if self.comm != None:
			self.comm.Disconnect()			
	def create_childrunner(self):
		import tempfile
		tmp = tempfile.NamedTemporaryFile(suffix='.py', prefix='childrunner', dir='./', delete=False)
		tmp.writelines([
			'from mpimap import start_child\r\n',
			'start_child()'])
		self.childrunner = tmp.name
		tmp.close()
	def __call__(self, function, iterable, *args, **kwargs):		
		self.comm.bcast((function, iterable, args, kwargs), root=MPI.ROOT)
		tmp = self.comm.gather(None, root=MPI.ROOT)
		rank = len(tmp)
		result = []
		for i in range(len(iterable)):
			index = divmod(i,rank)
			result.append(tmp[index[1]][index[0]])
		return result		
	
class MapperChild():
	def __init__(self):
		self.comm = MPI.Comm.Get_parent()
		self.size = self.comm.Get_size()
		self.rank = self.comm.Get_rank()
	def start(self):		
		while True:			
			args = self.comm.bcast(None, root=0)
			if args == None:
				return self.close()
			self.comm.gather(self(args[0], args[1], *args[2], **args[3]), root=0)
	def close(self):
		if self.comm != None:
			self.comm.Disconnect()
	def __call__(self, function, iterable, *args, **kwargs):
		return map(lambda x: function(x, *args, **kwargs), iterable[slice(self.rank, len(iterable), self.size)])

def start_child():
	child = MapperChild()
	#try:
	child.start()
	#except:
	#	child.comm.Abort()
#		raise