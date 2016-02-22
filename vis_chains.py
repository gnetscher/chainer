import numpy as np
import matplotlib.pyplot as plt
import chain as ch
from chainer_utils import utils as cu
from easydict import EasyDict as edict
import pickle
import os

class VisImBBox(ch.ChainObject):
	_consumer_ = [tuple]
	_producer_ = [None]
	def __init__(self, prms={}):
		dArgs = edict()
		dArgs.ax = None
		dArgs.drawOpts = {'color': 'r', 'linewidth': 3}
		prms     = cu.get_defaults(prms, dArgs, True)
		ch.ChainObject.__init__(self, prms)
		plt.ion()
		if prms.ax is None:
			fig = plt.figure()
			self.prms_.ax = fig.add_subplot(111)

	def produce(self, ip):
		im, bbox = ip
		self.prms_.ax.cla()
		self.prms_.ax.imshow(im)
		for b in bbox:
			x1, y1, x2, y2, conf = np.floor(b)
			self.prms_.ax.plot([x1, x1], [y1, y2], **self.prms_.drawOpts)
			self.prms_.ax.plot([x1, x2], [y2, y2], **self.prms_.drawOpts)
			self.prms_.ax.plot([x2, x2], [y2, y1], **self.prms_.drawOpts)
			self.prms_.ax.plot([x2, x1], [y1, y1], **self.prms_.drawOpts)
		plt.tight_layout()
		plt.draw()
		plt.show()

##
#Consumes a pickle detection filepath and visualizes the detection
class VisDetections(ch.ChainObject):

	_consumer_ = [str]
	_producer_ = [None]

	def __init__(self, prms=None):
		# initialize with image directory
		self.basePath = prms
		ch.ChainObject.__init__(self, prms)

	def produce(self, ip):
		detectDict = pickle.load(open(ip, 'rb'))
		visIB = VisImBBox()
		for detected in detectDict:
			for frame in detectDict[detected]:
				# find image frame
				filename  = '{:06d}.jpg'.format(int(frame[0]))
				imagePath = os.path.join(self.basePath, filename)
				im = plt.imread(imagePath)

				# show objects within frame
				objMat   = frame[2]
				visIB.produce((im, objMat))

