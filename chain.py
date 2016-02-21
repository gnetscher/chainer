import numpy as np
import copy
import os
from os import path as osp
import scipy.misc as scm
import easydict
from easydict import EasyDict as edict
import collections as co
from mAP.mAP import evaluate_mAP

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
	def __init__(self, objList=[], opData=None):
		'''
			objList: a list of chains that need to be chained together
			opData : The outputs that are desired
							 eg: opData = [[1,0], [3,1]] means that output 0
							 of module 1 and output 1 of module 3 need to be
								returned. Remembering first module is number 0. 
		'''
		self.chainObjs_ = []
		if opData is None:
			opData = [(-1,0)]
		#DummyObject helps in easily getting the desired outputs
		DummyObj = (ChainObject(), opData)
		objList.append(DummyObj)
		N       = len(objList)
		self.N_ = N
		#ip2opIdx[i] is a list that stores to what modules the o/p
		#of ith module must be sent
		self.ip2opIdx_ = []
		#ips[i] is a list that stores inputs to ith module
		self.ips_      = []
		#ipN_[i] stores how many inputs the ith module requires
		self.ipN_      = np.ones((N,1)).astype(np.int)
		for n in range(N):
			self.ip2opIdx_.append([])
			self.ips_.append([])
		#Organize what input needs to be fed into what o/p
		for n,o in enumerate(objList):
			if type(o) == tuple:
				print (o)
				if len(o) == 1:
					obj  = o[0]
					conn = [(-1,0)]
				else:
					assert len(o) == 2, 'Format not recognized'
					obj, conn = o
					assert type(conn) == list, 'Connectivity must be list'	
			else:
				obj, conn = o, [(-1,0)]
			self.chainObjs_.append(obj)
			if n == 0:
				continue
			self.ipN_[n] = len(conn)
			for ipLoc, cn in enumerate(conn):
				modNum, ipNum = cn
				if modNum < 0:
					modNum = n + modNum
				assert modNum < n, 'Modules can only be chained in\
							feedforward chains'
				#Put the ipNum^{th} op of modNum^{th} module at 
				#ipLoc in the inputs of nth module 
				self.ip2opIdx_[modNum].append([n, ipNum, ipLoc]) 	
	
	def reset_ips(self):
		del self.ips_
		self.ips_ = []
		for n in self.ipN_:
			arr = [None] * int(n)
			self.ips_.append(arr)

	def produce(self, ip=None):
		self.reset_ips()
		ip = copy.deepcopy(ip)
		self.ips_[0] = [ip]
		for n, o in enumerate(self.chainObjs_):
			modIp = self.ips_[n]
			if len(modIp) == 1:
				op = o.produce(modIp[0])
			else:
				op = o.produce(modIp)
			for mi in self.ip2opIdx_[n]:
				#Determine the module that will take the current
				#output as the input
				modNum, ipNum, ipLoc = mi
				if type(op) == tuple:
					tmpOp = copy.deepcopy(op[ipNum])
				else:
					assert ipNum == 0, 'Previous module produces only 1 o/p'
					tmpOp = copy.deepcopy(op)
				self.ips_[modNum][ipLoc] = tmpOp
		return self.ips_[self.N_-1]

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

        return imlist


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
    def __init__(self, prms='~/vatic/vatic'):
        """
        :param prms: location of vatic directory
        """
        self.vaticPath = prms
        ChainObject.__init__(self, prms)

    def produce(self, ip):
        """
        Given vatic identifier string, create vatic output text and return file path
        :param ip: vatic identifier string
        :return: vatic output text file path
        """
        # turkic dump identifier -o output.txt --merge --merge-threshold 0.5
        basePath  = str(os.getcwd())
        vaticFile = 'vaticOutput{0}.txt'.format(ip)
        sysCall = '(' + \
                  'cd {0}; '.format(self.vaticPath) + \
                  'turkic dump {0} -o {1} --merge --merge-threshold 0.5; '.format(ip, vaticFile) + \
                  'mv {0} {1}'.format(vaticFile, basePath)  + \
                  ')'
        os.system(sysCall)
        return os.path.join(basePath, vaticFile)

##
# Consumes a vatic text file path and returns list [frame, [box coordinates], label, (attributes)]
class Txt2Labels(ChainObject):

    _consumer_ = [str]
    _producer_ = [list, set]
    def __init__(self, prms=None):
        """
        :param prms: a set containing which contents to include
        must be from {'box', 'label', 'attributes', 'occluded', 'lost', 'generated'}
        :return: list of lists containing vatic information ordered as
        [frameNumber, labelString, [xmin, ymin, xmax, ymax], attributesStringList,
        lostBool, occludedBool, generatedBool] with the appropriate arguments ommitted
                 set containing the desired contents (e.g., 'box', 'label', etc.)
        """
        if prms is None:
            self.returnSet = {'box', 'label', 'attributes'}
        else:
            self.returnSet = prms
        ChainObject.__init__(self, prms)

    def produce(self, ip):
        # read in from file
        vaticList = []
        with open(ip, 'r') as f:
            for line in f:
                inList = []
                row  = line.split()
                inList.append(int(row[5]))
                if 'label' in self.returnSet:
                    inList.append(row[9].strip('"'))
                if 'box' in self.returnSet:
                    inList.append([int(x) for x in row[1:5]])
                if 'attributes' in self.returnSet:
                    inList.append([x.strip('"') for x in row[10:]])
                if 'lost' in self.returnSet:
                    inList.append(bool(row[6]))
                if 'occluded' in self.returnSet:
                    inList.append(bool(row[7]))
                if 'generated' in self.returnSet:
                    inList.append(bool(row[8]))
                vaticList.append(inList)

        # vatic only includes one detection per line -- combine lines from the same frame into one line
        vaticList.sort(key = lambda x: x[0])
        outList = []
        oRow = []
        for i, vRow in enumerate(vaticList):
            if i==len(vaticList)-1 or vaticList[i][0] != vaticList[i+1][0]:
                outList.append(list(oRow[:]))
                oRow = []
            else:
                oRow.append(tuple(vRow))

        return outList

##
# Consumes ground truth list and prediction list and returns the mAP
class Labels2mAP(ChainObject):

    # [ground truth list , prediction list]
    _consumer_ = [list, list]
    _producer_ = [float]

    def __init__(self, prms='onfloor'):
        """
        :param prms: check logic to be used to determine accuracy
                     formed by combining VATIC strs with logical operators
                     VATIC str to determine truth value: 'label', 'onfloor', 'falling'
                     e.g, 'label' will determine mAP based on successfully matching the label
                          'onfloor and falling' will check that onfloor matches onfloor and falling matches falling
                          'onfloor or  falling' will check that the prediction is either onfloor or falling
                            when the prediction is either onfloor or falling
        :return: Labels2mAP object
        """
        self.checkWords = prms[0]
        if len(prms) > 1:
            self.checkLogic = prms[-1]
        else:
            self.checkLogic = None
        ChainObject.__init__(self, prms)

    def produce(self, ip):
        """
        Computer mAP (mean average precision) by comparing ground truth to prediction
        :param ip: ground truth list, prediction list
            ground truth is list of lists where each element is (frame, label, [xmin, ymin, xmax, ymax], [attributes])
            prediction   is list of lists where each element is (frame, label, [xmin, ymin, xmax, ymax], [attributes], confidence)
            Note: Even if there are no attributes, an empty list will be received from the previous chain object
            Note: input of the occluded, generated, and lost flags is not currently accepted
        :return: mAP
        """
        actual, predicted = ip
        assert len(actual) == len(predicted)

        # form ground truth list in desired format -- if any label or attribute matches the desired logic, mark the
        # create a summary output label; if not mark the label as absent
        gt = []
        for frame in actual:
            inList = []
            for row in frame:
                words2Check = [row[1]] + row[3]
                nLabel = ''
                for word in self.checkWords:
                    if word in words2Check:
                        nLabel += word
                    if self.checkLogic == 'and':
                        if word not in words2Check:
                            nLabel = 'absent'
                            break
                if not nLabel:
                    nLabel = 'absent'
                inList.append(tuple([nLabel] + [row[2]]))
            gt.append(inList)

        # same process for predictions but include confidence score
        pd = []
        for frame in predicted:
            inList = []
            for row in frame:
                words2Check = [row[1]] + row[3]
                nLabel = ''
                for word in self.checkWords:
                    if word in words2Check:
                        nLabel += word
                    if self.checkLogic == 'and':
                        if word not in words2Check:
                            nLabel = 'absent'
                            break
                if not nLabel:
                    nLabel = 'absent'
                inList.append(tuple([nLabel] + [row[2]] + [row[-1]]))
            pd.append(inList)

        mAP = evaluate_mAP(gt, pd)
        return mAP




