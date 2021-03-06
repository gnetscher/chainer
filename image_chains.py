import pdb
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
import pickle
import matplotlib.pyplot as plt

##
#Consumes and produces image data
class Im2Im(ch.ChainObject):
	'''
		Image as as np array
	'''
	_consumer_ = [int, float, np.uint8, np.ndarray]
	_producer_ = [int, float, np.uint8, np.ndarray]
	def __init__(self, prms=None):
		ch.ChainObject.__init__(self, prms)
		
	def produce(self, ip):
		yield copy.deepcopy(ip)

##
#Consumes image name and produces image
class File2Im(ch.ChainObject):
	_consumer_ = [str]
	_producer_ = [np.ndarray]
	def __init__(self, prms=None):
		ch.ChainObject.__init__(self, prms)

	def produce(self, ip):
		im = scm.imread(ip)
		yield im

##
#Consumes RGB image and converts to BGR
class RGB2BGR(ch.ChainObject):
	_consumer_ = [np.ndarray]
	_producer_ = [np.ndarray]
	def __init__(self, prms=None):
		ch.ChainObject.__init__(self, prms)

	def produce(self, ip):
		yield ip[:,:,[2,1,0]]

##
#Consumes a directory and produces an iterator over images recursively searching all folders
class DataDir2IterImNames(ch.ChainObjectIter):
	"""
			assumes all images are jpg or png
	"""
	_consumer_ = [str]
	_producer_ = [str]
	def __init__(self, prms=None):
		ch.ChainObjectIter.__init__(self, prms)

	def produce(self, ip=None):
		extns = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG']
		for dirpath, dirnames, filenames in os.walk(ip):
				for name in dirnames:
						head = str(dirpath) + '/' + str(name)
						for extn in extns:
								path = os.path.join(head, '*.' + extn)
								for img in glob.glob(path):
										yield img
				else:
					head = str(dirpath)
					for extn in extns:
						path = os.path.join(head, '*.' + extn)
						for img in glob.glob(path):
							yield img



##
##Consumes a directory name and produces an iterator over images	
class DataDir2IterIms(ch.ChainObject):
	_consumer_ = [str]
	_producer_ = [np.ndarray]
	def __init__(self, prms=[{}, {}]):
		ch.ChainObject.__init__(self, prms)
		oList = []
		oList.append(DataDir2IterImNames(self.prms_[0]))
		oList.append(File2Im(self.prms_[1]))
		self.chain_ = ch.Chainer(oList)

	def produce(self, ip=None):
		yield self.chain_.produce(ip)


##
#Consumes a video path str and saves frames to directory returning the directory path str.
class Video2Ims(ch.ChainObject):
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
		ch.ChainObject.__init__(self, self.prms_)

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
		yield op_dir

##
#Consumes a pickle detection filepath and produces a list of cropped images
class Detection2Ims(ch.ChainObject):

	_consumer_ = str
	_producer_ = [np.array]

	def __init__(self, prms=None):
		self.basePath = prms
		ch.ChainObject.__init__(self, prms)

	def produce(self, ip):
		detectDict = pickle.load(open(ip, 'rb'))
		for detected in detectDict:
			for frame in detectDict[detected]:
				# find image frame
				filename  = '{:06d}.jpg'.format(int(frame[0]))
				imagePath = os.path.join(self.basePath, filename)
				im = plt.imread(imagePath)

				# find objects within frame
				objMat   = frame[2]
				for row in objMat:
					xmin = round(row[0])
					ymin = round(row[1])
					xmax = round(row[2])
					ymax = round(row[3])
					cropdIm = im[ymin:ymax, xmin:xmax,:]
					yield cropdIm

