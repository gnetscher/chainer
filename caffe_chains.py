import chain as ch
import caffe
import config as cfg
import numpy as np
import copy
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms 
from easydict import EasyDict as edict

##
#Consumes image and produces detection bounding box
class Im2RCNNDet(ch.ChainObject):
	_consumer_ = [np.ndarray]
	_producer_ = [np.ndarray]
	def __init__(self, prms={}):
		prms = cfg.get_rcnn_prms(**prms)	
		ch.ChainObject.__init__(self, prms)
		self.setup_net()

	def setup_net(self):
		caffe.set_mode_gpu()
		caffe.set_device(0)
		netFiles  = cfg.get_caffe_net_files(self.prms_.netName)	
		self.net_ = caffe.Net(netFiles.deployFile, 
				netFiles.netFile, caffe.TEST)
		self.cls_ = cfg.dataset2classnames(self.prms_.trainDataSet) 
	
	def produce(self, ip):
		scores, bbox   = im_detect(self.net_, ip)
		#Find the top class for each box
		bestClass  = np.argmax(scores,axis=1)
		bestScore  = np.max(scores, axis=1)
		allDet     = edict()
		for cl in self.prms_.targetClass:	
			clsIdx = self.cls_.index(cl)
			#Get all the boxes that belong to the desired class
			idx    = bestClass == clsIdx
			clScore = bestScore[idx]
			clBox   = bbox[idx,:]
			#Sort the boxes by the score
			sortIdx  = np.argsort(-clScore)
			topK     = min(len(sortIdx), self.prms_.topK)
			sortIdx  = sortIdx[0:topK]
			#Get the desired output
			clScore = clScore[sortIdx]
			clBox   = clBox[sortIdx]
			clBox   = clBox[:, (clsIdx * 4):(clsIdx*4 + 4)]
			#Stack detections and perform NMS
			dets=np.hstack((clBox, clScore[:,np.newaxis])).astype(np.float32)
			keep = nms(dets, self.prms_.nmsThresh)
			dets = dets[keep, :]
			#Only keep detections with high confidence
			inds = np.where(dets[:, -1] >= self.prms_.confThresh)[0]
			allDet[cl]   = copy.deepcopy(dets[inds, :4])
		return ip, allDet

##
#Consumes image and produces detection of people
class Im2PersonDet(Im2RCNNDet):
	def __init__(self, prms={}):
		Im2RCNNDet.__init__(self, prms)

	def produce(self, ip):
		ip, allDet = super(Im2PersonDet, self).produce(ip)
		return ip, allDet['person']
		
