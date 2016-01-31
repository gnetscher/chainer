import numpy as np
import copy
import os
from os import path as osp
import scipy.misc as scm
import easydict
from easydict import EasyDict as edict
import collections as co
import glob

class ChainObject(object):
    '''
		Parent class for chaining 
		objects together
	'''	
    #consumer: what data format does it consume
    _consumer_ = [type(None)]
    #producer: what data format is produced
    _producer_ = [type(None)]
    def __init__(self, prms=None):
        self.prms_ = copy.deepcopy(prms)

    def _check_input(self, ip):
        assert(type(ip)) in self._consumer_, ('Object of type %s,\
		cannot be consumed' % type(ip))

    def get_producer_type(self):
        return copy.deepcopy(self._producer_)

    def get_consumer_type(self):
        return copy.deepcopy(self._consumer_)

    #The output of the instance
    def produce(self, ip=None):
        return ip


class Chainer(object):
	'''
		class that strings together chains
	'''
	def __init__(self, objList=[]):
		self.chainObjs_ = objList

	def produce(self, ip):
		ip = copy.deepcopy(ip)
		for o in self.chainObjs_:
			ip = o.produce(ip)
		return ip

##
#Consumes and produces image data
class ImData(ChainObject):
    '''
		Image as as np array
	'''
    _consumer_ = [int, float, np.uint8, np.ndarray]
    _producer_ = [int, float, np.uint8, np.ndarray]
    def __init__(self, prms=None):
        ChainObject.__init__(self, prms)
		
    def produce(self, ip):
        return copy.deepcopy(ip)

##
#Consumes image name and produces image
class ImDataFile(ChainObject):
    _consumer_ = [str]
    _producer_ = [np.ndarray]
    def __init__(self, prms=None):
        ChainObject.__init__(self, prms)

    def produce(self, ip):
        im = scm.imread(ip)
        return im

##
#Consumes RGB image and converts to BGR
class RGB2BGR(ChainObject):
	_consumer_ = [np.ndarray]
	_producer_ = [np.ndarray]
	def __init__(self, prms=None):
		ChainObject.__init__(self, prms)

	def produce(self, ip):
		return ip[:,:,[2,1,0]]


##
#Consumes a directory and produces an iterator over images
class ImDataDir(ChainObject):
    """
        assumes all images are jpg or png
    """
    _consumer_ = [str]
    _producer_ = [int, float, np.uint8]
    def __init__(self, prms=None):
        ChainObject.__init__(self, prms)

    def produce(self, ip):
        imlist = []
        extns = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG']

        for extn in extns:
            path = os.path.join(ip, '*.' + extn)
            for img in glob.glob(path):
                im = scm.imread(img)
                imlist.append(im)

        return iter(imlist)


##
#Consumes a video and saves frames to directory.
class Video2Ims(ChainObject):
    """
        mostly borrowed from vatic script
    """
    _consumer_ = [str]
    _producer_ = [str]
    def __init__(self, prms=None):
        ChainObject.__init__(self, prms)

    def produce(self, ip):
        pass
        # try:
        #     os.makedirs(args.output)
        # except:
        #     pass
        # sequence = ffmpeg.extract(args.video)
        # try:
        #     for frame, image in enumerate(sequence):
        #         if frame % 100 == 0:
        #             print ("Decoding frames {0} to {1}"
        #                 .format(frame, frame + 100))
        #         if not args.no_resize:
        #             image.thumbnail((args.width, args.height), Image.BILINEAR)
        #         path = Video.getframepath(frame, args.output)
        #         try:
        #             image.save(path)
        #         except IOError:
        #             os.makedirs(os.path.dirname(path))
        #             image.save(path)
        # except:
        #     if not args.no_cleanup:
        #         print "Aborted. Cleaning up..."
        #         shutil.rmtree(args.output)
        #     raise


