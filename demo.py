import numpy as np
import chain as ch
from os import path as osp

def run_rcnn():
    imPath = '/mnt/HardDrive/datatry/outimg_Angle1Lighting1'
    imName = osp.join(imPath, '0.jpg')
    imProd = ch.ImDataFile()
    return imProd

def run_test():
    imPath = 'try/Falls_Angle1Lighting1.mp4'
    # imName = osp.join(imPath, '0.jpg')
    # imProd = ch.ImDataFile()

    imProd = ch.Video2Ims(prms={'op_dir': 'op'})
    print imProd.produce(imPath)


if __name__ == '__main__':
    run_test()