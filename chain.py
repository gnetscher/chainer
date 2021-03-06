import numpy as np
import copy
import os
from os import path as osp
import scipy.misc as scm
import easydict
from easydict import EasyDict as edict
import collections as co
import types
import itertools

ITER_STOP_SYMBOL = 'ITER_STOP_SYMBOL'
STOP_SYMBOLS     = [ITER_STOP_SYMBOL]

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
		yield ip

##
#Chain Object subclass of type iterator
class ChainObjectIter(ChainObject):
	'''
		Iterators return a stopsymbol if the iterator
		has been exhausted
	'''
	_consumer_ = [type(None)]
	_producer_ = [type(None)]

	def __init__(self, prms=None):
		ChainObject.__init__(self, prms)
		self.stopsymbol_ = ITER_STOP_SYMBOL


class Chainer(object):
	'''
		class that strings together chains
	'''
	def __init__(self, objList=[], opData=None):
		'''
			objList: a list of chains that need to be chained together
			opData : The outputs that are desired
							 eg: opData = [[1,0], [3,1]] means that output 0
							 of module 1 and output 1 of module 3 need to be
								returned. Remembering first module is number 0. 
		'''
		self.isValid_   = True
		self.chainObjs_ = []
		if opData is None:
			opData = [(-1,0)]
		#DummyObject helps in easily getting the desired outputs
		DummyObj = (ChainObject(), opData)
		objList.append(DummyObj)
		N       = len(objList)
		self.N_ = N
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
				if modNum < 0:
					modNum = n + modNum
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

	#Is the chain producing valid outputs
	def is_valid(self):
		return self.isValid_

	#Check if stop symbol has been produced or not
	def _check_stop(self, op):
		if type(op) in [tuple, list]:
			for opt in op:
				if type(opt) in [str] and opt in STOP_SYMBOLS:
					self.isValid_ = False
			else:
				if type(op) in [str]:
					if op in STOP_SYMBOLS:
						self.isValid_ = False


	def produce(self, ip=None):
		self.reset_ips()
		ip = copy.deepcopy(ip)
		self.ips_[0] = [ip]
		for n, o in enumerate(self.chainObjs_):
			if n == len(self.chainObjs_) - 1:
				# not sure why, but final iteration just passes through chain object base class
				break
			modIp = self.ips_[n][0]
			if not self.isValid_:
				return None
			if isinstance(modIp, types.GeneratorType) or isinstance(modIp, itertools.chain):
				op = ()
				# expect modIp to be a generator
				for inp in modIp:
					nOp = o.produce(inp)
					op = itertools.chain(op, nOp)
			else:
				# not generator
				op = o.produce(modIp)

			for mi in self.ip2opIdx_[n]:
				#Determine the module that will take the current
				#output as the input
				modNum, ipNum, ipLoc = mi
				if type(op) == tuple:
					tmpOp = copy.deepcopy(op[ipNum])
				elif isinstance(op, types.GeneratorType) or isinstance(op, itertools.chain):
					tmpOp = op  # if really need copy, use: tmpOp, op = itertools.tee(op)
				else:
					try:
						assert ipNum == 0, 'Previous module produces only 1 o/p'
					except:
						print ("ERROR")
						print modNum, ipNum, ipLoc, n
						return
					tmpOp = copy.deepcopy(op)
				self.ips_[modNum][ipLoc] = tmpOp
		return list(self.ips_[self.N_-1][0])
