import numpy as np
import chain as ch
import caffe_chains as cc
from os import path as osp

#Sample image used in the demo
def get_sample_imname():
	imPath = '/mnt/HardDrive/data/try/outimg_Angle1Lighting1'	
	imName = osp.join(imPath, '0.jpg')
	return imName

#Testing the ImDataFile Module
def image_reader():
	imName = get_sample_imname()
	imProd = ch.ImDataFile()
	im     = imProd.produce(imName)
	return imProd

def run_rcnn():
	imName = get_sample_imname()
	imProd = ch.ImDataFile()
	bgr    = ch.RGB2BGR()
	rcnn   = cc.Im2RCNNDet()
	chain  = ch.Chainer([imProd, bgr, rcnn])
	allDet = chain.produce(imName) 
	im     = imProd.produce(imName)
	return im, allDet

def run_test():
    imPath = 'try/'
    # imName = osp.join(imPath, '0.jpg')
    # imProd = ch.ImDataFile()

    imProd = ch.ImDataDir()
    iter_ = imProd.produce(imPath)
    for it in iter_:
        print '~~~~~~ IMAGE ~~~~~~'
        #print it

if __name__ == '__main__':
    run_test()
