import numpy as np
from collections import defaultdict, Mapping, Sequence

def evaluate_mAP(gt, pd, k=2, overlapThreshold=0.5):
    """
    Evaluate the prediction against the ground truth using mAP
    The idea is to find all sets of overlapping boxes, then evaluate the mean average precision
    :param gt: ground truth list of lists where each element is (label, [xmin, ymin, xmax, ymax]);'
                        'rows refer to frames and columns to objects within frames'
    :param pd: prediction list of lists where each element is (label, [xmin, ymin, xmax, ymax], confidence);'
                        'rows refer to frames and columns to objects within frames
    :return:   mAP metric on dataset
    """
    if isinstance(gt, Mapping) and isinstance(pd, Mapping):
        gtList = [d.items() for d in gt.values()]
        pdList = [list((key, d[key][0], d[key][1]) for key in d.keys()) for d in pd.values()]
    elif isinstance(gt, Sequence) and isinstance(pd, Sequence):
        gtList = gt
        pdList = pd
    else:
        raise TypeError('Inputs should be lists of lists where each element is (label, [xmin, ymin, xmax, ymax], confidence);'
                        'rows refer to frames and columns to objects within frames')

    ## make assignments of bounding boxes based on overlap
    # find all bounding boxes with overlap at desired threshold
    gMatchListOuter = []
    pMatchListOuter = []
    for i, frame in enumerate(gtList):
        gtq = frame[:]
        pdq = pdList[i][:]
        while gtq:
            # check all overlapping sets containing at least one ground truth value
            gtCurrent  = gtq.pop()
            gtLabel    = gtCurrent[0]
            gtbox      = gtCurrent[1]
            pMatchList = []
            for j1, pCurrent in enumerate(pdList[i]):
                pLabel = pCurrent[0]
                pbox   = pCurrent[1]
                ol     = calc_overlap(pbox, gtbox)
                if ol > overlapThreshold:
                    pMatchList.append(pCurrent)
                    if pCurrent in pdq:
                        pdq.remove(pCurrent)
            pMatchList.sort(key=lambda match: match[2], reverse=True)
            pMatchList = [match[0] for match in pMatchList]

            gMatchList = [gtLabel]
            for j2, gCurrent in enumerate(gtq[:]):
                gLabel = gCurrent[0]
                gbox   = gCurrent[1]
                ol     = calc_overlap(gtbox, gbox)
                if ol > overlapThreshold:
                    # part of the same set of overlapping bounding boxes
                    gMatchList.append(gLabel)
                    gtq.remove(gCurrent)

            gMatchListOuter.append(gMatchList)
            pMatchListOuter.append(pMatchList)

        while pdq:
            # check remaining sets containing no ground truth value
            pdCurrent  = pdq.pop()
            pdLabel    = pdCurrent[0]
            pdbox      = pdCurrent[1]
            pMatchList = [pdLabel]
            for j, pCurrent in enumerate(pdq[:]):
                pLabel = pCurrent[0]
                pbox   = pCurrent[1]
                ol     = calc_overlap(pdbox, pbox)
                if ol > overlapThreshold:
                    # part of the same set of overlapping bounding boxes
                    pMatchList.append(pLabel)
                    pdq.remove(pCurrent)

            gMatchListOuter.append([])
            pMatchListOuter.append(pMatchList)

    # calculate mAP based on overlapping regions
    return calc_mAP(gMatchListOuter, pMatchListOuter, k=k)

def calc_overlap(box1, box2):
    """
    Given two boxes, return the overlap as the intersection over the union
    # TODO: improve efficiency
    :param box1: [xmin, ymin, xmax, ymax]
    :param box2: [xmin, ymin, xmax, ymax]
    :return: overlap
    """
    # quick check that an overlap exists
    if box1[0] > box2[2] or box2[0] > box1[2] or box1[1] > box2[3] or box2[1] > box1[3]:
        return 0.0

    # inner rect
    xsw = min(box1[2], box2[2]) - max(box1[0], box2[0])
    ysw = min(box1[3], box2[3]) - max(box1[1], box2[1])
    ina = xsw * ysw

    # existing box areas
    b1a = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2a = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # overlap
    return float(ina) / (b1a + b2a - ina)

def calc_mAP(gtl, pdl, k=1):
    """
    Computes the mean average precision at k
    Takes a ground truth list and prediction list where the rows of each are matched bounding boxes
    By default, one prediction is allowed per ground truth (ie, equivalent to mAP@1)
    :param gtl: ground truth list of labels
    :param pdl: prediction list of labels
    :param k  : the maximum number of predicted elements per ground truth element
    :return: mAP score
    """
    return np.mean([calc_AP(g,p,k) for g,p in zip(gtl, pdl)])


def calc_AP(gtl, pdl, k=1):
    """
    Computes the average precision at k
    :param gtl: ground truth list of labels
    :param pdl: prediction list of labels
    :param k  : the maximum number of predicted elements per ground truth element
    :return: average precision
    """
    if len(gtl)>k:
        pdl = pdl[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(pdl):
        if p in gtl and p not in pdl[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not gtl:
        return 0.0

    return score / min(len(gtl), k)

def evaluate_intuitive(gt, pd, overlapThreshold=0.5, outputMatching=False):
    """
    Evaluate the prediction against the ground truth using intuitive metric.
    In an overlapping set, match ground truth labels to predictions with the same label.
    (e.g., if gt = [person, dog] and pd = [dog, person] in an overlapping set, match the labels to each other
    :param gt: ground truth dict as {frame: {label: [xmin, ymin, xmax, ymax]}}
    :param pd: prediction dict as {frame: {label: ([xmin, ymin, xmax, ymax], confidence)}}
    :return:   mAP metric on dataset
    """
    ## make assignments of bounding boxes based on overlap
    # find all bounding boxes with overlap at desired threshold
    outerDic = {}
    for frame in gt:
        matchedDict = defaultdict(bool) # contains the predictions we've already matched
        gtq = gt[frame].keys() # store keys as queue
        innerDic = {}
        while gtq:
            gtLabel = gtq.pop()
            gbox        = gt[frame][gtLabel]
            pMatchList  = []
            for pdLabel in pd[frame]:
                pbox = pd[frame][pdLabel][0]
                ol   = calc_overlap(pbox, gbox)
                if ol > overlapThreshold:
                    pMatchList.append([pdLabel, pd[frame][pdLabel]])

            if pMatchList:
                # choose the bounding box with highest confidence as prediction
                pMatchList.sort(key=lambda match: match[1][1], reverse=True)
                pMatch = pMatchList[0][0]
                #pMatch = pMatchList[np.argmax([match[1][1] for match in pMatchList])]
                if matchedDict[pMatch]:
                    ## resolve conflict between 2 ground truth values wanting the same prediction

                    # find original ground truth value matching this prediction
                    gPrev = None
                    for gp, pp in innerDic.items():
                        if pp == pMatch:
                            gPrev = gp
                    if not gPrev:
                        raise ValueError('gPrev should not be None if a match occurred')

                    # check which match is better
                    if pMatch in innerDic and pMatch == innerDic[pMatch]:
                        # if current label matches current prediction, then keep and take next best c-score
                        if len(pMatchList) > 1:
                            innerDic[gtLabel] = pMatchList[1][0]
                            matchedDict[pMatchList[1][0]] = True
                        # if no next best prediction, then don't make any new match
                    else:
                        # form list of possible other matches
                        gMatchList = []
                        for gMatch in gt[frame]:
                            gboxn = gt[frame][gMatch]
                            pboxn = pd[frame][pMatch][0]
                            ol   = calc_overlap(pboxn, gboxn)
                            if ol > overlapThreshold:
                                gMatchList.append(gMatch)

                        # check if prediction matches any ground truth label in possible set
                        for g in gMatchList:
                            if g == pMatch: # note this will only be true at most once by uniqueness of keys
                                innerDic.pop(gPrev)
                                gtq.append(gPrev)
                                innerDic[g] = pMatch
                                # note matchedDict[pMatch] is already true for this prediction

                        # if not, then choose one with greater overlap
                        if not matchedDict[pMatch]:
                            olp = calc_overlap(gt[frame][gPrev], pd[frame][pMatch][0])
                            oln = calc_overlap(gt[frame][gtLabel], pd[frame][pMatch][0])
                            if olp > oln:
                                # keep current matching and take next best c-score
                                if len(pMatchList) > 1:
                                    innerDic[gtLabel] = pMatchList[1][0]
                                    matchedDict[pMatchList[1][0]] = True
                            else:
                                innerDic.pop(gPrev)
                                gtq.append(gPrev)
                                innerDic[gtLabel] = pMatch


                else:
                    matchedDict[pMatch] = True
                    innerDic[gtLabel]   = pMatch

        for gk in gt[frame]:
            if gk not in innerDic:
                innerDic[gk] = ''
        outerDic[frame] = innerDic

    # outerDic format {frame: {gtlabel: pdlabel}}

    # calculate mAP based on matched labels
    outMap = calc_mAP_from_dict(outerDic)

    # return desired output
    if outputMatching:
        return outMap, outerDic
    else:
        return outMap

def calc_mAP_from_dict(d, k=1):
    """
    Computes the mean average precision at k
    By default, one prediction is allowed per ground truth (ie, equivalent to mAP@1)
    :param d : dictionary in format {frame: {gtlabel: pdlabel}}
    :param k : the maximum number of predicted elements per ground truth element
    :return: mAP score
    """
    matchList = []
    for k in d:
        for g, p in d[k].iteritems():
            matchList.append([g, p])
    return np.mean([calc_AP(g, p, k) for g, p in matchList])

if __name__ == '__main__':
    # 2d dict with image frame as key: label as key: bounding, confidence score
    # bounding box as [xmin, ymin, xmax, ymax]
    # Note with this 2d dict structure frame and label must be unique (i.e., can't be two person labels in 1 frame)
    gt = {0: {'person': [0, 0, 10, 10]},
          1: {'person': [0, 0, 11, 11], 'dog': [20, 20, 30, 30]},
          2: {'person': [0, 0, 12, 12], 'dog': [0, 0, 13, 13]},
          3: {'dog': [0, 0, 12, 12], 'person': [0, 0, 13, 13]}}

    pd = {0: {'person': ([0, 0, 10, 10], 0.70), 'dog': ([6, 6, 11, 11], 0.75), 'cat': ([0, 0, 10, 10], 0.65)},
          1: {'person': ([0, 0, 11, 11], 0.71)},
          2: {'person': ([0, 0, 12, 12], 0.72)},
          3: {'person': ([0, 0, 13, 13], 0.71), 'dog': ([0, 0, 12, 12], 0.72)}}
    print evaluate_intuitive(gt, pd)


