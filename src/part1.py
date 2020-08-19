from collections import defaultdict
from itertools import islice
from pprint import pprint
import numpy as np
import os

def smoothedEstimation(tags,words,labelWords,k):
    emissionPrbability = {}
    for tag in labelWords:
        emissionPrbability[tag] = {}
        # labelWords[tag]['#UNK#'] = 0
        for word in list(labelWords[tag]):
            if word not in words:
                labelWords[tag]['#UNK#'] += labelWords[tag].pop(word)
            elif words[word] < k:
                labelWords[tag]['#UNK#'] += labelWords[tag].pop(word)
                del words[word]
            else:
                emissionPrbability[tag][word] = labelWords[tag][word] / tags[tag]
        # emissionPrbability[tag]['#UNK#'] = labelWords[tag]['#UNK#'] / tags[tag]
    trainFile= list(words)
    return trainFile,emissionPrbability


def calculate_trans_parameter(filePath):
    tags = {}
    transitionTag = {}
    transitionProbability = {}
    _preT = ''
    _newT = 'START'
    for line in open(filePath, encoding='utf-8', mode='r'):
        _preT = _newT if (_newT != 'STOP') else 'START'
        segmentedLine = line.rstrip()

        if segmentedLine:
            segmentedLine = segmentedLine.rsplit(' ', 1)
            _newT = segmentedLine[1]
        else:
            _newT = 'STOP'
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


def calculate_feature(emission_parameter, transition_parameter):
    feature_dic = defaultdict()

    # for word in emission_parameter.keys():
    #     for idx, tag in enumerate(emission_parameter[word][0]):
    #         feature_dic["emission:" + tag + '+' + word] = np.log2(emission_parameter[word][1][idx])
    for tag in emission_parameter.keys():
        for word in emission_parameter[tag]:
            feature_dic["emission:" + tag + "+" +word] = np.log2(emission_parameter[tag][word])
    for word in transition_parameter.keys():
        for word2 in transition_parameter[word]:
            feature_dic["transition:" + word + '+' + word2] = np.log2(transition_parameter[word][word2])

    return feature_dic


def dataProcessing(filePath):
    tags = {}
    words = {}
    labelWords = {}

    for line in open(filePath, encoding='utf-8', mode='r'):
        segmentedLine = line.strip()
        if segmentedLine:
            word, tag = lineCut(segmentedLine)
            if word not in words:
                words[word] = 1
            else:
                words[word] += 1
            if tag not in tags:
                tags[tag] = 1
                labelWords[tag] = {word: 1}
            else:
                tags[tag] += 1
                if word not in labelWords[tag]:
                    labelWords[tag][word] = 1
                else:
                    labelWords[tag][word] += 1

    return tags, words, labelWords


def lineCut(segmentedLine):
    segmentedLine = segmentedLine.rsplit(' ', 1)
    word = segmentedLine[0]
    tag = segmentedLine[1]
    return word,tag


def generate_data(path_train):
    train_lines = list(filter(None, open(path_train).read().splitlines()))
    train_data = [line.split() for line in train_lines]
    train_x = [line[0] for line in train_data if line]
    train_y = [line[1] for line in train_data if line]

    # # emission_para = calculate_emission_parameter(train_data, train_x, train_y)
    # tags,words,labelWords= dataProcessing(path_train)
    # trainFile,emission_para = smoothedEstimation(tags,words,labelWords,k=-1)
    # transition_para = calculate_trans_parameter(path_train)
    # # print(emission_para)
    # feature = calculate_feature(emission_para, transition_para)
    # print(feature)
    return train_data, train_x, train_y


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


path_train = os.path.join(os.path.dirname( __file__ ),"..","data","partial","train")
train_data, train_x, train_y = generate_data(path_train)

tags, words, labelWords = dataProcessing(path_train)
trainFile, emission_para = smoothedEstimation(tags, words, labelWords, k=-1)
transition_para = calculate_trans_parameter(path_train)
feature_dict = calculate_feature(emission_para, transition_para)


if __name__ == '__main__':
    n_items = take(5, feature_dict.items())
    pprint(n_items)
