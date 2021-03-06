import numpy as np
import chain as ch
import caffe_chains as cc
import image_chains as imc
import misc_chains as mc
import data_chains as dc
import vis_chains as vc
import metric_chains as mec
import vatic_chains as vac
from os import path as osp
import pickle

#Sample image used in the demo
def get_sample_imname():
	imPath = '/mnt/HardDrive/data/try/outimg_Angle1Lighting1'
	imName = osp.join(imPath, '0.jpg')
	return imName

#Testing the File2Im Module
def image_reader():

	imName = get_sample_imname()
	imProd = imc.File2Im()
	im     = imProd.produce(imName)
	return imProd

def run_rcnn():
	imName = get_sample_imname()
	imProd = imc.File2Im()
	bgr    = imc.RGB2BGR()
	rcnn   = cc.Im2RCNNDet()
	chain  = ch.Chainer([imProd, bgr, rcnn])
	im, allDet = chain.produce(imName)
	return im, allDet


def run_rcnn_iter():
	dataSrc   = dc.GetDataDir()
	src2Im    = imc.DataDir2IterIms()
	bgr       = imc.RGB2BGR()
	rcnn   = cc.Im2PersonDet()
	vis    = vc.VisImBBox()
	chain  = ch.Chainer([dataSrc, src2Im, bgr, rcnn,
					 (vis, [[1,0],[-1,0]])])
	return chain


def save_rcnn_op(dataFn, opName):
	opFile   = osp.join('tmp', opName)
	dataSrc  = dataFn
	src2Name = imc.DataDir2IterImNames()
	name2Im  = imc.File2Im()
	bgr      = imc.RGB2BGR()
	rcnn     = cc.Im2PersonDet()
	imKey    = mc.File2SplitLast()
	chain    = ch.Chainer([dataSrc, src2Name, name2Im, bgr, \
			 rcnn, (imKey, [(1,0)])], opData=[(-1,0),(-2,0)])
	count = 0
	data  = []
	while True:
		op = chain.produce()
		if op is None:
			break
		frame, bbox = op	
		data.append([frame, 'person', bbox])
		count += 1
		print (count)
	pickle.dump({'person_det': data}, open(opFile, 'w'))


def save_rcnn_vatic():
	dataFn = dc.GetDataDirVatic()
	opName = 'vatic_person_det.pkl' 
	save_rcnn_op(dataFn, opName)

def save_rcnn_demo():
	dataFn = dc.GetDataDirDemo()
	opName = 'demo_person_det.pkl'
	save_rcnn_op(dataFn, opName)

def save_rcnn_mpii():
	dataFn = dc.GetDataDirMPII()
	opName = 'mpii_person_det.pkl'
	save_rcnn_op(dataFn, opName)



def run_test():
	# vidPath = 'try/Falls_Angle1Lighting1.mp4'
	# imProd  = ch.Video2Ims()
	# imPath  = imProd.produce(vidPath)
	# itProd  = ch.ImDataDir()
	# list_   = itProd.produce(imPath)
	# for i, item in enumerate(list_):
	#     print '~~~~Image {0}~~~~'.format(i)
	#     print item

	# vaticID = 'Angle1Lighting1'
	# vaticProd = ch.Ims2Txt()
	# txtPath = vaticProd.produce(vaticID)
	txtPath   = './try/output_Angle1Lighting1.txt'
	labelProd = vac.Txt2Labels()
	actual    = labelProd.produce(txtPath)
	mapProd   = mec.Labels2mAP([['onfloor', 'falling'], 'or'])
	# test vatic output against itself by giving each ground truth a confidence score of 1
	predicted = []
	for frame in actual:
		inFrame = []
		for object in frame:
			inObject = object[:]
			inObject += (1.0,)
			inFrame.append(inObject)
		predicted.append(inFrame)

	testOut   = mapProd.produce((actual, predicted))
	print testOut

def test_cropping():
	picklePath = './tmp/person_det.pkl'
	imageDir   = './tmp/'
	cropProd   = imc.Detection2Ims(imageDir)
	print cropProd.produce(picklePath)

def test_vis_cropping():
	picklePath = './tmp/person_det.pkl'
	imageDir   = './tmp/'
	visProd    = vc.VisDetections(imageDir)
	visProd.produce(picklePath)

def test_it_handling():
	# to test fix for issue #6
	src2Name = imc.DataDir2IterImNames()
	name2Im  = imc.File2Im()
	bgr      = imc.RGB2BGR()
	chain    = ch.Chainer([src2Name, name2Im, bgr])
	for x in chain.produce('./tmp/'):
		print x


if __name__ == '__main__':
	test_it_handling()
