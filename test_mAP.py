import unittest
from mAP.mAP import evaluate_intuitive, evaluate_mAP
from vatic_chains import Txt2Labels
from metric_chains import Labels2mAP
import os

class TestAveragePrecision(unittest.TestCase):

	def setUp(self):
		self.groundTruth = \
			{0: {'person': [0, 0, 10, 10]},
			1: {'person': [0, 0, 11, 11], 'dog': [20, 20, 30, 30]},
			2: {'person': [0, 0, 12, 12], 'dog': [0, 0, 13, 13]},
			3: {'dog': [0, 0, 12, 12], 'person': [0, 0, 13, 13]}}

		self.predicted   = \
			{0: {'person': ([0, 0, 10, 10], 0.70), 'dog': ([6, 6, 11, 11], 0.75), 'cat': ([0, 0, 10, 10], 0.65)},
			1: {'person': ([0, 0, 11, 11], 0.71)},
			2: {'person': ([0, 0, 12, 12], 0.72)},
			3: {'person': ([0, 0, 13, 13], 0.71), 'dog': ([0, 0, 12, 12], 0.72)}}

		self.expectedIntuitiveOutDict = \
			{0: {'person': 'person'},
			1: {'person': 'person', 'dog': ''},
			2: {'person': 'person', 'dog': ''},
			3: {'person': 'person', 'dog': 'dog'}}

		self.expectedIntuitiveMetric = 0.714285714286

		self.actualIntuitiveMetric, self.actualIntuitiveOutDict = evaluate_intuitive(self.groundTruth, self.predicted, outputMatching=True)

		# hand calulcation of MAP@2
		# AP = sum(P@i * R@i - R@(i-1)) calculating for each separate set of overlapping boxes
		# frame 0: (1*1) + (1/2 * 0) = 1
		#        : 0
		# frame 1: 1
		#        : 0
		# frame 2: 0.5
		# frame 3: 1
		# mAP = 3.5/6 = 0.5833
		self.expectedMAP = 0.5833

		self.actualMAP = evaluate_mAP(self.groundTruth, self.predicted, k=2, overlapThreshold=0.5)

	def test__intuitive_matching(self):
		self.assertEquals(self.expectedIntuitiveOutDict, self.actualIntuitiveOutDict)

	def test_intuitive_metric(self):
		self.assertAlmostEquals(self.expectedIntuitiveMetric, self.actualIntuitiveMetric)

	def test_mAP_metric(self):
		# simple suite of tests
		self.assertAlmostEquals(self.expectedMAP, self.actualMAP, places=3)

	def test_mAP_VATIC(self):
		BASE_DIR  = os.path.dirname(__file__)
		txtPath   = os.path.join(BASE_DIR, 'try/output_Angle1Lighting1.txt')
		labelProd = Txt2Labels()
		actual    = labelProd.produce(txtPath)
		mapProd   = Labels2mAP([['onfloor', 'falling'], 'or'])
		# test vatic output against itself by giving each ground truth a confidence score of 1
		predicted = []
		for frame in actual:
			inFrame = []
			for object in frame:
				inObject = object[:]
				inObject += (1.0,)
				inFrame.append(inObject)
			predicted.append(inFrame)

		testOut   = mapProd.produce((actual, predicted))
		self.assertAlmostEquals(testOut, 1.0)

if __name__ == '__main__':
	unittest.main()