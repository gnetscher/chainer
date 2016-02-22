from chain import ChainObject
from mAP.mAP import evaluate_mAP

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



