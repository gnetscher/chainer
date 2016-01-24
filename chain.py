import numpy as np
import copy
import os
from os import path as osp
import scipy.misc as scm
import easydict
from easydict import EasyDict as edict
import collections as co

class ChainObject(object):
	'''
		Parent class for chaining 
		objects together
	'''	
	#consumer: what data format does it consumes
	_consumer_ = [type(None)]
	#producer: what data format is produced
	_producer_ = [type(None)]
	def __init__(self, prms=None):
		self.prms_ = copy.deepcopy(prms)

	def _check_input(self, ip):
		assert(type(ip)) in self._consumer_, ('Object of type %s,\
		cannot be consumed' % type(ip))

	def get_producer_type(self):
		return copy.deepcopy(self._producer_)

	def get_consumer_type(self):
		return copy.deepcopy(self._consumer_)

	#The output of the instance
	def produce(self, ip=None):
		return ip

##
#Consumes and produces image data
class ImData(ChainObject):
	'''
		Image as as np array
	'''
	_consumer_ = [int, float, np.uint8, np.ndarray]
	_producer_ = [int, float, np.uint8, np.ndarray]
	def __init__(self, prms=None):
		ChainObject.__init__(self, prms)
		
	def produce(self, ip):
		return copy.deepcopy(ip)
		
##
#Consumes image name and produces image
class ImDataFile(ChainObject):
	_consumer_ = [str]
	_producer_ = [np.ndarray]
	def __init__(self, prms=None):
		ChainObject.__init__(self, prms)

	def produce(self, ip):
		im = scm.imread(ip)
		return im

##
#Consumes a directory and produces an iterator over images
class ImDataDir(ChainObject):
	_consumer_ = [str]
	_producer_ = [int, float, np.uint8]
	def __init__(self, prms=None):
		ChainObject.__init__(self, prms)

##
#Consumes a video and saves frames to directory.
class Video2Ims(ChainObject):
	_consumer_ = [str]
	_producer_ = [str]
	def __init__(self, prms=None):
		ChainObject.__init__(self, prms)
	#### To Complete ######

##
#Consumes image and produces detection bounding box
class Im2RCNNDet(ChainObject):
	_consumer_ = [np.ndarray]
	_producer_ = [np.ndarray]
	def __init__(self, prms=None):
		ChainObject.__init__(self, prms)
		
	def produce(self, ip):
		pass	
