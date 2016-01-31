from easydict import EasyDict as edict
from pkg.pycaffe_utils import other_utils as ou 
from os import path as osp
import os

def get_basic_paths():
	paths = edict()
	paths.base    = edict()
	paths.base.dr =  '/mnt/HardDrive/common'
	return paths


def caffe_model_paths():
	paths = get_basic_paths()	
	paths.caffemodel = edict()
	paths.caffemodel.dr = osp.join(paths.base.dr, 'caffe_models')
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
	if netName == 'vgg16':
		oPrms.netFile = 'VGG16.caffemodel'
		oPrms.defFile = ''
		baseDr        = paths.caffemodel.imagenet.dr
	elif netName == 'vgg16-pascal-rcnn':
		oPrms.netFile = 'VGG16_faster_rcnn_final.caffemodel'
		oPrms.deployFile = 'test.prototxt'
		oPrms.solFile    = 'solver.prototxt' 
		oPrms.defFile    = 'train.prototxt' 
		baseDr           = osp.join(paths.caffemodel.fasterrcnn.dr,'VGG16')
	elif netName == 'zf-pascal-rcnn':
		oPrms.netFile = 'ZF_faster_rcnn_final.caffemodel'
		oPrms.defFile = ''
		baseDr           = paths.caffemodel.fasterrcnn.dr
	for k in oPrms.keys():
		oPrms[k] = osp.join(baseDr, oPrms[k])
	return oPrms

##
#Default class names for some datasets
def dataset2classnames(dataSet='pascal'):
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
	dArgs.trainDataSet = 'pascal'
	#The net to be used 
	dArgs.netName    = 'vgg16-pascal-rcnn'
	return dArgs	
