from collections import defaultdict
from itertools import islice
from pprint import pprint
import numpy as np
import os


class Feature():
    def __init__(self, path_train):
        self.path_train = path_train
        self.tags, self.words, self.label_words = self.data_processing()
        self.emission_parameter= self.calculate_emiss_parameter()
        self.transition_parameter = self.calculate_trans_parameter()
        self.feature_dict = self.calculate_feature()

    def calculate_emiss_parameter(self):
        '''
        a function to estimate the following emission probabilities
        based on the training set.
        '''
        emission_prbability = {}
        for tag in self.label_words:
            emission_prbability[tag] = {}
            self.label_words[tag]['#UNK#'] = 0
            for word in list(self.label_words[tag]):
                if word not in self.words:
                    self.label_words[tag]['#UNK#'] += self.label_words[tag].pop(word)
                elif self.words[word] < 3:
                    self.label_words[tag]['#UNK#'] += self.label_words[tag].pop(word)
                    del self.words[word]
                else:
                    emission_prbability[tag][word] = self.label_words[tag][word] / self.tags[tag]
            emission_prbability[tag]['#UNK#'] = self.label_words[tag]['#UNK#'] / self.tags[tag]
        return emission_prbability

    def calculate_trans_parameter(self):
        '''
        a function to estimate the transition probabilities.
        '''
        tags = {}
        transition_tag = {}
        transition_probability = {}
        _preT = ''
        _newT = 'START'
        for line in open(self.path_train, encoding='utf-8', mode='r'):
            _preT = _newT if (_newT != 'STOP') else 'START'
            segmented_line = line.rstrip()

            if segmented_line:
                segmented_line = segmented_line.rsplit(' ', 1)
                _newT = segmented_line[1]
            else:
                _newT = 'STOP'
            if _preT not in tags:
                tags[_preT] = 1
                transition_tag[_preT] = {_newT: 1}
            else:
                tags[_preT] += 1
                if _newT not in transition_tag[_preT]:
                    transition_tag[_preT][_newT] = 1
                else:
                    transition_tag[_preT][_newT] += 1

        for tag in transition_tag:
            transition_probability[tag] = {}
            for transition in transition_tag[tag]:
                transition_probability[tag][transition] = np.log2(transition_tag[tag][transition] / tags[tag])

        return transition_probability

    def calculate_feature(self):
        '''
        create feature dictionary based on emission and transition probability.
        '''
        feature_dic = defaultdict()

        for tag in self.emission_parameter.keys():
            for word in self.emission_parameter[tag]:
                feature_dic["emission:" + tag + "+" +word] = self.emission_parameter[tag][word]
        for word in self.transition_parameter.keys():
            for word2 in self.transition_parameter[word]:
                feature_dic["transition:" + word + '+' + word2] = self.transition_parameter[word][word2]

        return feature_dic

    def data_processing(self):
        tags = {}
        words = {}
        label_words = {}

        for line in open(self.path_train, encoding='utf-8', mode='r'):
            segmented_line = line.strip()
            if segmented_line:
                word, tag = self.line_cut(segmented_line)
                if word not in words:
                    words[word] = 1
                else:
                    words[word] += 1
                if tag not in tags:
                    tags[tag] = 1
                    label_words[tag] = {word: 1}
                else:
                    tags[tag] += 1
                    if word not in label_words[tag]:
                        label_words[tag][word] = 1
                    else:
                        label_words[tag][word] += 1

        return tags, words, label_words

    def line_cut(self, segmented_line):
        segmented_line = segmented_line.rsplit(' ', 1)
        word = segmented_line[0]
        tag = segmented_line[1]
        return word, tag


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


if __name__ == '__main__':
    feature = Feature(os.path.join(os.path.dirname( __file__ ),"..", "data", "partial", "train"))
    feature_dict = feature.feature_dict
    n_items = take(5, feature_dict.items())
    pprint(n_items)
