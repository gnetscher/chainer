from chain import ChainObject
import os

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
		yield os.path.join(basePath, vaticFile)

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

		yield outList
