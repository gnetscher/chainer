from easydict import EasyDict as edict

def get_caffe_net_details(nPrms):
	oPrms = edict()
	if nPrms['net'] = 'vgg16':
		oPrms.netFile = 'VGG16.caffemodel'
		oPrms.defFile = ''
	elif nPrms['net'] = 'vgg16-rcnn':
		oPrms.netFile = 'VGG16_faster_rcnn_final.caffemodel'
		oPrms.defFile = '' 
	elif nPrms['net'] = 'zf-rcnn':
		oPrms.netFile = 'ZF_faster_rcnn_final.caffemodel'
		oPrms.defFile = ''


def get_rcnn_prms():
		
