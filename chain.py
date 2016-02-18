import numpy as np
import copy
import os
from os import path as osp
import scipy.misc as scm
import easydict
from easydict import EasyDict as edict
import collections as co

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
				#Redo the computation if True
				#This is useful when data is cached
				#and can be loaded
				self.redo_ = False

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

	def produce(self, ip=None):
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
#Consumes a directory and produces an iterator over images recursively searching all folders
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

        for dirpath, dirnames, filenames in os.walk(ip):
            for name in dirnames:
                head = str(dirpath) + '/' + str(name)
                for extn in extns:
                    path = os.path.join(head, '*.' + extn)
                    for img in glob.glob(path):
                        im = scm.imread(img)
                        imlist.append(im)

        return iter(imlist)


##
#Consumes a video path str and saves frames to directory returning the directory path str.
class Video2Ims(ChainObject):
    """
        output directory can optionally be given as a parameter as {'op_dir': <path>}
    """

    _consumer_ = [str]
    _producer_ = [str]
    def __init__(self, prms=None):
        """
        :param prms: directory of parameters in this case containing only a possible output dir
                     e.g., {'op_dir': '~/Desktop/'}
                    if None, the input directory is used when produce is called
        :return:
        """
        try:
            op_dir = prms['op_dir']
        except:
            # wait to specify upon learning input video directory
            op_dir = None
        print op_dir
        self.prms_ = {'op_dir': op_dir}
        ChainObject.__init__(self, self.prms_)

    def produce(self, ip):
        def getframepath(frame, base = None):
            l1 = frame / 10000
            l2 = frame / 100
            path = "{0}/{1}/{2}.jpg".format(l1, l2, frame)
            if base is not None:
                path = "{0}/{1}".format(base, path)
            return path

        if self.prms_['op_dir'] is None:
            op_dir = os.path.dirname(ip)
        else:
            op_dir = self.prms_['op_dir']
        try:
            os.makedirs(op_dir)
        except:
            pass
        sequence = ffmpeg.extract(ip)
        try:
            for frame, image in enumerate(sequence):
                if frame % 100 == 0:
                    print ("Decoding frames {0} to {1}"
                        .format(frame, frame + 100))
                path = getframepath(frame, op_dir)
                try:
                    image.save(path)
                except IOError:
                    os.makedirs(os.path.dirname(path))
                    image.save(path)
        except:
            print "ffmpeg may not be installed"
            print "Aborted. Cleaning up..."
            shutil.rmtree(op_dir)
            raise
        return op_dir

##
#  given vatic identifier string create vatic text file path
class Ims2Txt(ChainObject):

    _consumer_ = [str]
    _producer_ = [str]
    def __init__(self, prms=None):
        ChainObject.__init__(self, prms)

    def produce(self, ip, vaticPath='~/vatic/vatic'):
        # turkic dump identifier -o output.txt --merge --merge-threshold 0.5
        basePath  = str(os.getcwd())
        vaticFile = 'vaticOutput{0}.txt'.format(ip)
        sysCall = '(' + \
                  'cd {0}; '.format(vaticPath) + \
                  'turkic dump {0} -o {1} --merge --merge-threshold 0.5; '.format(ip, vaticFile) + \
                  'mv {0} {1}'.format(vaticFile, basePath)  + \
                  ')'
        os.system(sysCall)
        return os.path.join(basePath, vaticFile)

##
# Consumes a vatic text file path and returns iterator over [frame, [box coordinates], (attributes)]
class Txt2Labels(ChainObject):

    _consumer_ = [str]
    _producer_ = [np.uint8, [np.uint8, np.uint8, np.uint8, np.uint8], str, [str]]
    def __init__(self, prms=None):
        """
        :param prms: a set containing which contents to include
        must be from {'box', 'label', 'attributes', 'occluded', 'lost', 'generated}
        :return: iterator over list of lists containing vatic information ordered as
        [frameNumber, [xmin, ymin, xmax, ymax], lostBool,
            occludedBool, generatedBool, labelString, attributesStringList]
        with the appropriate arguments ommitted
        """
        if prms is None:
            self.returnSet = {'box', 'label', 'attributes'}
        else:
            self.returnSet = prms
        ChainObject.__init__(self, prms)

    def produce(self, ip):
        outList = []
        with open(ip, 'r') as f:
            for line in f:
                inList = []
                row  = line.split()
                inList.append(int(row[5]))
                if 'box' in self.returnSet:
                    inList.append([int(x) for x in row[1:5]])
                if 'lost' in self.returnSet:
                    inList.append(bool(row[6]))
                if 'occluded' in self.returnSet:
                    inList.append(bool(row[7]))
                if 'generated' in self.returnSet:
                    inList.append(bool(row[8]))
                if 'label' in self.returnSet:
                    inList.append(row[9].strip('"'))
                if 'attributes' in self.returnSet:
                    inList.append([x.strip('"') for x in row[10:]])
                outList.append(inList)

        return iter(outList)
