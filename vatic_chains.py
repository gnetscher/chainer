import chain as ch
import os

##
#  given vatic identifier string create vatic text file path
class Ims2Txt(ch.ChainObject):
    _consumer_ = [str]
    _producer_ = [str]
    def __init__(self, prms=None):
        ch.ChainObject.__init__(self, prms)

    def produce(self, ip):
        # turkic dump identifier -o output.txt --merge --merge-threshold 0.5
        basePath  = str(os.getcwd())
        vaticFile = 'vaticOutput{0}.txt'.format(ip)
        print vaticFile, ' ', basePath
        sysCall = '(' + \
                  'cd ~/vatic/vatic; ' + \
                  'turkic dump {0} -o {1} --merge --merge-threshold 0.5;'.format(ip, vaticFile) + \
                  'mv {0} {1}'.format(vaticFile, basePath)  + \
                  ')'
        os.system(sysCall)
        return os.path.join(basePath, vaticFile)

##
# Consumes a vatic text file path and returns iterator over [frame, [box coordinates], (attributes)]
class Txt2Labels(ch.ChainObject):

    _consumer_ = [str]
    _producer_ = [np.uint8, [np.uint8, np.uint8, np.uint8, np.uint8], str, [str]]
    def __init__(self, prms=None):
        """
        :param prms: a set containing which contents to include
        must be from {'box', 'label', 'attributes', 'occluded', 'lost', 'generated}
        :return: iterator over list of lists containing vatic information ordered as
        [frameNumber, [xmin, ymin, xmax, ymax], lostBool,
            occludedBool, generatedBool, labelString, attributesStringList]
        with the appropriate arguments ommitted
        """
        if prms is None:
            self.returnSet = {'box', 'label', 'attributes'}
        else:
            self.returnSet = prms
        ch.ChainObject.__init__(self, prms)

    def produce(self, ip):
        outList = []
        with open(ip, 'r') as f:
            for line in f:
                inList = []
                row  = line.split()
                inList.append(int(row[5]))
                if 'box' in self.returnSet:
                    inList.append([int(x) for x in row[1:5]])
                if 'lost' in self.returnSet:
                    inList.append(bool(row[6]))
                if 'occluded' in self.returnSet:
                    inList.append(bool(row[7]))
                if 'generated' in self.returnSet:
                    inList.append(bool(row[8]))
                if 'label' in self.returnSet:
                    inList.append(row[9].strip('"'))
                if 'attributes' in self.returnSet:
                    inList.append([x.strip('"') for x in row[10:]])
                outList.append(inList)

        return iter(outList)
