from collections import defaultdict
from itertools import islice
from pprint import pprint
import numpy as np
import os

class Feature():
    def __init__(self, path_train):
        self.path_train = path_train
        self.train_data, self.train_x, self.train_y = self.generate_data(self.path_train)
        self.tags, self.words, self.labelWords = self.dataProcessing(self.path_train)
        self.emission_para = self.calculate_emiss_parameter(self.tags, self.words, self.labelWords)
        self.transition_para = self.calculate_trans_parameter(self.path_train)
        self.feature_dict = self.calculate_feature(self.emission_para, self.transition_para)

    def calculate_emiss_parameter(self, tags, words, labelWords):
        emissionPrbability = {}
        for tag in labelWords:
            emissionPrbability[tag] = {}
            for word in list(labelWords[tag]):
                emissionPrbability[tag][word] = labelWords[tag][word] / tags[tag]
        return emissionPrbability

    def calculate_trans_parameter(self, filePath):
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

        for tag in transitionTag:
            transitionProbability[tag] = {}
            for transition in transitionTag[tag]:
                transitionProbability[tag][transition] = transitionTag[tag][transition] / tags[tag]

        return transitionProbability

    def calculate_feature(self, emission_parameter, transition_parameter):
        feature_dic = defaultdict()

        for tag in emission_parameter.keys():
            for word in emission_parameter[tag]:
                feature_dic["emission:" + tag + "+" +word] = np.log2(emission_parameter[tag][word])
        for word in transition_parameter.keys():
            for word2 in transition_parameter[word]:
                feature_dic["transition:" + word + '+' + word2] = np.log2(transition_parameter[word][word2])

        return feature_dic
    def generate_data(self, path_train):
        train_lines = list(filter(None, open(path_train).read().splitlines()))

        train_data = [line.split() for line in train_lines]
        train_x = [line[0] for line in train_data if line]
        train_y = [line[1] for line in train_data if line]

        return train_data, train_x, train_y

    def dataProcessing(self, filePath):
        tags = {}
        words = {}
        labelWords = {}

        for line in open(filePath, encoding='utf-8', mode='r'):
            segmentedLine = line.strip()
            if segmentedLine:
                word, tag = self.lineCut(segmentedLine)
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

    def lineCut(self, segmentedLine):
        segmentedLine = segmentedLine.rsplit(' ', 1)
        word = segmentedLine[0]
        tag = segmentedLine[1]
        return word, tag


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


if __name__ == '__main__':
    feature = Feature(os.path.join(os.path.dirname( __file__ ),"..","data","partial","train"))
    feature_dict = feature.feature_dict
    n_items = take(5, feature_dict.items())
    pprint(n_items)
