import chain as ch
from chainer_utils import utils as cu
from easydict import EasyDict as edict
import socket

def get_datadirs():
	pth = edict()
	hostname = socket.gethostname()
	if hostname in ['nestsense']:
		pth.falls = '/home/nestsense/data/Falls/%s/vatic/outputimg/outimg_raw_%s'
	else:
		raise Exception('HostName %s not recognized' % hostname)
	return pth
	
##
#Returns the datadirectory
class GetDataDir(ch.ChainObject):
	_consumer_ = [None]
	_producer_ = [str]
	def __init__(self, prms={}):
		dArgs = edict()
		dArgs.folderName    = 'nicks-house'
		dArgs.subFolderName = 'Angle1Lighting1'
		prms = cu.get_defaults(prms, dArgs, True)
		ch.ChainObject.__init__(self, prms)
		pths = get_datadirs()
		self.prms_.dirName = pths.falls % (self.prms_.folderName,
					 self.prms_.subFolderName) 

	def produce(self, ip=None):
		return self.prms_.dirName
