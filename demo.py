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
    # vidPath = 'try/Falls_Angle1Lighting1.mp4'
    # imProd  = ch.Video2Ims()
    # imPath  = imProd.produce(vidPath)
    # itProd  = ch.ImDataDir()
    # list_   = itProd.produce(imPath)
    # for i, item in enumerate(list_):
    #     print '~~~~Image {0}~~~~'.format(i)
    #     print item
    vaticID = 'Angle1Lighting1'
    vaticProd = ch.Ims2Txt()
    txtPath = vaticProd.produce(vaticID)
    labelProd = ch.Txt2Labels()
    test =  labelProd.produce(txtPath)
    for a in test:
        print a



if __name__ == '__main__':
    run_test()
