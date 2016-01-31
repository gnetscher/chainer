import chain as ch
import caffe
import config as cfg
import numpy as np

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
		pass
