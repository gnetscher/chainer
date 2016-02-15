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
	def __init__(self, objList=[], opBlobs=None):
		'''
			objList: a list of chains that need to be chained together
			opData : what 
		'''
		self.chainObjs_ = []
		N  = len(objList)
		#ip2opIdx[i] is a list that stores to what modules the o/p
		#of ith module must be sent
		self.ip2opIdx_ = []
		#ips[i] is a list that stores inputs to ith module
		self.ips_      = []
		#ipN_[i] stores how many inputs the ith module requires
		self.ipN_      = np.ones((N,1)).astype(np.int)
		for n in range(N):
			self.ip2opIdx_.append([])
			self.ips_.append([])
		#Organize what input needs to be fed into what o/p
		for n,o in enumerate(objList):
			if type(o) == tuple:
				print (o)
				if len(o) == 1:
					obj  = o[0]
					conn = [(-1,0)]
				else:
					assert len(o) == 2, 'Format not recognized'
					obj, conn = o
					assert type(conn) == list, 'Connectivity must be list'	
			else:
				obj, conn = o, [(-1,0)]
			self.chainObjs_.append(obj)
			if n == 0:
				continue
			self.ipN_[n] = len(conn)
			for ipLoc, cn in enumerate(conn):
				modNum, ipNum = cn
				if modNum == -1:
					self.ip2opIdx_[n-1].append([n, ipNum, ipLoc])
				else:
					assert modNum < n, 'Modules can only be chained in\
                feedforward chains'
					#Put the ipNum^{th} op of modNum^{th} module at 
					#ipLoc in the inputs of nth module 
					self.ip2opIdx_[modNum].append([n, ipNum, ipLoc]) 	
	
	def reset_ips(self):
		del self.ips_
		self.ips_ = []
		for n in self.ipN_:
			arr = [None] * int(n)
			self.ips_.append(arr)

	def produce(self, ip=None):
		self.reset_ips()
		ip = copy.deepcopy(ip)
		self.ips_[0] = [ip]
		for n, o in enumerate(self.chainObjs_):
			modIp = self.ips_[n]
			op    = o.produce(*modIp)
			for mi in self.ip2opIdx_[n]:
				#Determine the module that will take the current
				#output as the input
				modNum, ipNum, ipLoc = mi
				if type(op) == tuple:
					tmpOp = copy.deepcopy(op[ipNum])
				else:
					assert ipNum == 0, 'Previous module produces only 1 o/p'
					tmpOp = copy.deepcopy(op)
				self.ips_[modNum][ipLoc] = tmpOp 
		return op

