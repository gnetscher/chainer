import numpy as np
import chain as ch
import caffe_chains as cc
import image_chains as imc
import data_chains as dc
from os import path as osp

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
	rcnn   = cc.Im2RCNNDet()
	chain  = ch.Chainer([dataSrc, src2Im, bgr, rcnn])
	return chain

def run_test():
    vidPath = 'try/Falls_Angle1Lighting1.mp4'
    imProd  = ch.Video2Ims()
    imPath  = imProd.produce(vidPath)
    itProd  = ch.ImDataDir()
    list_   = itProd.produce(imPath)
    for i, item in enumerate(list_):
        print '~~~~Image {0}~~~~'.format(i)
        print item


if __name__ == '__main__':
    run_test()
