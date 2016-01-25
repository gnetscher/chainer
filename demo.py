import numpy as np
import chain as ch
from os import path as osp

def run_rcnn():
	imPath = '/mnt/HardDrive/data/try/outimg_Angle1Lighting1'	
	imName = osp.join(imPath, '0.jpg')
	imProd = ch.ImDataFile()
	im     = imProd.produce(imName)
	return imProd
