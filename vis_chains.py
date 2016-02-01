import numpy as np
import matplotlib.pyplot as plt
import chain as ch
from chainer_utils import utils as cu
from easydict import EasyDict as edict

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
			x1, y1, x2, y2 = np.floor(b)
			self.prms_.ax.plot([x1, x1], [y1, y2], **self.prms_.drawOpts)
			self.prms_.ax.plot([x1, x2], [y2, y2], **self.prms_.drawOpts)
			self.prms_.ax.plot([x2, x2], [y2, y1], **self.prms_.drawOpts)
			self.prms_.ax.plot([x2, x1], [y1, y1], **self.prms_.drawOpts)
		plt.tight_layout()
		plt.draw()
		plt.show()	
				
