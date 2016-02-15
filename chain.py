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
    #consumer: what data format does it consume
    _consumer_ = [type(None)]
    #producer: what data format is produced
    _producer_ = [type(None)]
    def __init__(self, prms=None):
        self.prms_ = copy.deepcopy(prms)
				#Redo the computation if True
				#This is useful when data is cached
				#and can be loaded
				self.redo_ = False

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


class Chainer(object):
	'''
		class that strings together chains
	'''
	def __init__(self, objList=[]):
		self.chainObjs_ = []
		N  = len(objList)
		#ip2opIdx[i] is a list that stores to what modules the o/p
		#of ith module must be sent
		self.ip2opIdx_ = []
		#ips[i] is a list that stores inputs to ith module
		self.ips_      = []
		for n in range(N):
			self.ip2opIdx_.append([])
			self.ips_.append([])
		#Organize what input needs to be fed into what o/p
		for n,o in enumerate(objList):
			if type(o) == tuple:
				if len(o) == 1:
					obj  = o[0]
					conn = ((-1,0))
				else:
					assert len(o) == 2, 'Format not recognized'
					self.chainObjs_.append(o[0])
					obj, conn = o
					assert type(conn) == tuple, 'Connectivity must be tuple'	
			if n == 0:
				continue
			for cn in conn:
				modNum, ipNum = cn
				if modNum == -1:
					self.ip2opIdx_[n-1].append((n, ipNum))
				else:
					assert modNum < n, 'Modules can only be chained in\
                feedforward chains'
					self.ip2opIdx_[modNum].append((n, ipNum)) 	
				

	def produce(self, ip=None):
		ip = copy.deepcopy(ip)
		for o in self.chainObjs_:
			ip = o.produce(ip)
		return ip

