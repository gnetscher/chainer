import numpy as np
import copy
import os
from os import path as osp
import scipy.misc as scm

class ChainObject(object):
	'''
		Parent class for chaining 
		objects together
	'''	
	#consumer: what data format does it consumes
	_consumer_ = [type(None)]
	#producer: what data format is produced
	_producer_ = [type(None)]
	def __init__(self, prevObj=None):
		self.prev_ = prevObj

	def _check_consumer(self):
		assert(type(self.prev_)) in self._consumer_, ('Object of type %s,\
		cannot be consumed' % type(self.prev_))

	def get_producer_type(self):
		return copy.deepcopy(self._producer_)

	def get_consumer_type(self):
		return copy.deepcopy(self._consumer_)

	#The output of the instance
	def produce(self):
		return None

##
#Consumes and produces image data
class ImData(ChainObject):
	'''
		Image as as np array
	'''
	_consumer_ = [int, float, np.uint8, np.ndarray]
	_producer_ = [int, float, np.uint8, np.ndarray]
	def __init__(self, prevObj):
		ChainObject.__init__(self, prevObj)
		self._check_consumer()
		
	def produce(self):
		return copy.deepcopy(self.prev_)
		
##
#Consumes image name and produces image
class ImDataFile(ChainObject):
	_consumer_ = [str]
	_producer_ = [np.ndarray]
	def __init__(self, prevObj):
		ChainObject.__init__(self, prevObj)
		self._check_consumer()

	def produce(self):
		im = scm.imread(self.prev_)
		return im

##
#Consumes a directory and produces an iterator over images
class ImDataDir(ChainObject):
	_consumer_ = [str]
	_producer_ = [int, float, np.uint8]
	def __init__(self, prevObj):
		ChainObject.__init__(self, prevObj)
		self._check_consumer()

##
#Consumes a video and saves frames to directory.
class Video2Ims(ChainObject):
	_consumer_ = [str]
	_producer_ = [str]
	def __init__(self, prevObj):
		ChainObject.__init__(self, prevObj)
		self._check_consumer()

	
