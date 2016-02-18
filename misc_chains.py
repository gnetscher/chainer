import chain as ch
import glob
import shutil
import ffmpeg
import numpy as np
import copy
import os
from os import path as osp
import scipy.misc as scm
import easydict
from easydict import EasyDict as edict

##
#Consumes a file name and returns the last split of the filename
#without the extension
class File2SplitLast(ch.ChainObject):
    _consumer_ = [str]
    _producer_ = [str]
    def __init__(self, prms=None):
        ch.ChainObject.__init__(self, prms)

    def produce(self, ip):
        return osp.basename(ip).split('.')[0]
       


