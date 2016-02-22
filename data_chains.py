import chain as ch
from chainer_utils import utils as cu
from easydict import EasyDict as edict
import socket
from os import path as osp

def get_datadirs():
	pth = edict()
	hostname = socket.gethostname()
	if hostname in ['nestsense']:
		pth.vatic = '/home/nestsense/data/Falls/%s/vatic/outputimg/outimg_raw_%s'
		pth.dsets = '/mnt/HardDrive/nestdata/datasets/'
	else:
		raise Exception('HostName %s not recognized' % hostname)
	return pth
	
##
#Returns the datadirectory
class GetDataDir(ch.ChainObject):
	_consumer_ = [None]
	_producer_ = [str]
	def __init__(self, prms=edict()):
		dArgs = edict()
		ch.ChainObject.__init__(self, prms)
	
	def produce(self, ip=None):
		return self.prms_.dirName

##
#Return data directory for vatic
class GetDataDirVatic(GetDataDir):
	def __init__(self, prms=edict()):
		dArgs.folderName    = 'nicks-house'
		dArgs.subFolderName = 'Angle1Lighting1'
		prms = cu.get_defaults(prms, dArgs, True)
		GetDataDir.__init__(self, prms)
		pths = get_datadirs()
		self.prms_.dirName = pths.vatic % (self.prms_.folderName,
					 self.prms_.subFolderName)

##
#Return data directory for MPII
class GetDataDirMPII(GetDataDir):
 def __init__(self, prms=edict()):
	GetDataDir.__init__(self, prms)
	pths = get_datadirs()
	self.prms_.dirName = osp.join(pths.dsets, 'mpii', 'images')


##
#Return data directory for demo experiments
class GetDataDirDemo(GetDataDir):
 def __init__(self, prms=edict()):
	GetDataDir.__init__(self, prms)
	pths = get_datadirs()
	self.prms_.dirName = osp.join(pths.dsets, 'demo', 'images')

