from easydict import EasyDict as edict
from pkg.pycaffe_utils import other_utils as ou 
from os import paths as osp
import os

def get_basic_paths():
	paths = edict()
	paths.base    = edict()
	paths.base.dr =  '/mnt/HardDrive/common'
	return paths


def caffe_model_paths():
	paths = get_basic_paths()	
	paths.caffemodel = edict()
	paths.caffemodel.dr = osp.join(paths.basedr, 'caffe_models')
	#Faster-rcnn model
	paths.caffemodel.fasterrcnn    = edict()
	paths.caffemodel.fasterrcnn.dr = osp.join(paths.caffemodel.dr, 
		'faster_rcnn_models')
	#imagenet model dr
	paths.caffemodel.imagenet    = edict()
	paths.caffemodel.imagenet.dr = osp.join(paths.caffemodel.dr,
		'imagenet_models') 
	return paths 


def get_caffe_net_files(netName):
	paths = caffe_model_paths()
	oPrms = edict()
	#Caffe Model File
	oPrms.netFile = ''
	#Deploy prototxt
	oPrms.deployFile = ''
	#Solver prototxt
	oPrms.solverFile = ''
	#Train file
	oPrms.defFile    = ''
	if netName = 'vgg16':
		oPrms.netFile = 'VGG16.caffemodel'
		oPrms.defFile = ''
		baseDr        = paths.caffemodel.imagenet.dr
	elif netName = 'vgg16-rcnn':
		oPrms.netFile = 'VGG16_faster_rcnn_final.caffemodel'
		oPrms.deployFile = 'test.prototxt'
		oPrms.solFile    = 'solver.prototxt' 
		oPrms.defFile    = 'train.prototxt' 
		baseDr           = paths.caffemodel.fasterrcnn.dr
	elif netName = 'zf-rcnn':
		oPrms.netFile = 'ZF_faster_rcnn_final.caffemodel'
		oPrms.defFile = ''
	return oPrms

##
#Default class names for some datasets
def get_class_names(dataSet='pascal'):
	if dataSet == 'pascal':
		cls = list(('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor'))	
	else:
		raise Exception('Dataset %s not found' % dataSet)
	return cls

##
#Default arguments for rcnn
def get_rcnn_prms(**kwargs):
	dArgs = edict()
	#NMS
	dArgs.nmsThresh  = 0.3
	#Detection Confidence
	dArgs.confThresh = 0.8
	#What classnames was the detector trained on.
	dArgs.detDataSet = 'pascal'
	return dArgs	
