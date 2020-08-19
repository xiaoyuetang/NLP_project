import argparse
import tqdm


# def lineCut(segmentedLine):
#     segmentedLine = segmentedLine.rsplit(' ', 1)
#     word = segmentedLine[0]
#     tag = segmentedLine[1]
#     return word, tag


# def dataProcessing(filePath):
#     tags = {}
#     words = {}
#     labelWords = {}
#
#     for line in open(filePath, encoding='utf-8', mode='r'):
#         segmentedLine = line.strip()
#         if segmentedLine:
#             word, tag = lineCut(segmentedLine)
#             if word not in words:
#                 words[word] = 1
#             else:
#                 words[word] += 1
#             if tag not in tags:
#                 tags[tag] = 1
#                 labelWords[tag] = {word: 1}
#             else:
#                 tags[tag] += 1
#                 if word not in labelWords[tag]:
#                     labelWords[tag][word] = 1
#                 else:
#                     labelWords[tag][word] += 1
#
#     return tags, words, labelWords

#
# def smoothedEstimation(tags, words, labelWords, k):
#     emissionPrbability = {}
#     for tag in labelWords:
#         emissionPrbability[tag] = {}
#         labelWords[tag]['#UNK#'] = 0
#         for word in list(labelWords[tag]):
#             if word not in words:
#                 labelWords[tag]['#UNK#'] += labelWords[tag].pop(word)
#             elif words[word] < k:
#                 labelWords[tag]['#UNK#'] += labelWords[tag].pop(word)
#                 del words[word]
#             else:
#                 emissionPrbability[tag][word] = labelWords[tag][word] / tags[tag]
#         emissionPrbability[tag]['#UNK#'] = labelWords[tag]['#UNK#'] / tags[tag]
#     trainFile = list(words)
#     return trainFile, emissionPrbability


def estimateTransition(filePath):
    tags = {}
    transitionTag = {}
    transitionProbability = {}
    _preT = ''
    _newT = '##START##'
    for line in open(filePath, encoding='utf-8', mode='r'):
        _preT = _newT if (_newT != '##STOP##') else '##START##'
        segmentedLine = line.rstrip()

        if segmentedLine:
            segmentedLine = segmentedLine.rsplit(' ', 1)
            _newT = segmentedLine[1]
        else:
            _newT = '##STOP##'
        if _preT not in tags:
            tags[_preT] = 1
            transitionTag[_preT] = {_newT: 1}
        else:
            tags[_preT] += 1
            if _newT not in transitionTag[_preT]:
                transitionTag[_preT][_newT] = 1
            else:
                transitionTag[_preT][_newT] += 1

    calculateTransitionProbability(transitionProbability, transitionTag, tags)

    return transitionProbability


def calculateTransitionProbability(transitionProbability, transitionTag, tags):
    for tag in transitionTag:
        transitionProbability[tag] = {}
        for transition in transitionTag[tag]:
            transitionProbability[tag][transition] = transitionTag[tag][transition] / tags[tag]


# def getEstimate(sequence, trainingSet, label, k=0):
#     if sequence[k] in trainingSet:
#         if sequence[k] in emissionEstimates[label]:
#             emission = emissionEstimates[label][sequence[k]]
#         else:
#             emission = 0.0
#     else:
#         emission = emissionEstimates[label]['#UNK#']
#     return emission




if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-d', type=str, dest='dataset', help='Dataset to run script over', required=True)

    # args = parser.parse_args()

    # trainFilePath = './%s/train' % (args.dataset)
    # inputTestFilePath = './%s/dev.in' % (args.dataset)
    # outputTestFilePath = './%s/dev.p3.out' % (args.dataset)

    trainFilePath = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\train"
    inputTestFilePath = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\dev.in"
    outputTestFilePath = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\dev.out"

    transitionEstimates = estimateTransition(trainFilePath)
    # tags, words, labelWords = dataProcessing(trainFilePath)
    # trainFile, emissionEstimates = smoothedEstimation(tags, words, labelWords, k=3)
    # writeToFile(inputTestFilePath, trainFile, emissionEstimates, transitionEstimates, outputTestFilePath)
    print(transitionEstimates)