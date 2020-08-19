from collections import defaultdict
from pprint import pprint
import numpy as np
import os

from part1 import Feature


class CRF():
    def __init__(self, path_train):
        self.feature = Feature(path_train)
        self.feature_dict = self.feature.feature_dict
        self.tags, self.words, self.labelWords = self.feature.tags, self.feature.words, self.feature.labelWords
        self.labelCounts = len(self.tags)
        self.score_dict = self.calculate_score()

    def calculate_score(self):
        '''
        a function to calculate the score for a given pair of input and output
        sequence pair (x_seq, y_seq), based on the above-mentioned features
        and weights used in Part 1.

        '''
        score_dict = dict()
        for feature, weight in self.feature_dict.items():
            if feature not in score_dict:
                score_dict[feature] = weight
            else:
                score_dict[feature] += weight

        return score_dict

    def viterbi(self, word, potential_table,layer):
        '''
        perform decoding using the Viterbi algorithm to find the most
        probable output sequence y∗ for an input sequence x.
        '''
        pi = np.zeros((layer, self.labelCounts))
        yTable = np.zeros((layer, self.labelCounts), dtype='int64')
        #Initialization
        t = 0
        for index in range(self.labelCounts):
            pi[t, index] = potential_table[t][StartingIndex, index]
        #Recursive
        for t in range(1, layer):
            for index in range(1, self.labelCounts):
                maxPi = 0
                bestLabel = 0
                for prev_label_id in range(1, self.labelCounts):
                    value = pi[t-1, prev_label_id] * potential_table[t][prev_label_id, index]
                    if value > maxPi:
                        maxPi = value
                        bestLabel = prev_label_id
                pi[t, index] = maxPi
                yTable[t, index] = bestLabel
        sequence = list()
        next_label = pi[layer-1].argmax()
        sequence.append(next_label)
        #Final Decode
        for t in range(layer-1, -1, -1):
            next_label = yTable[t, next_label]
            sequence.append(next_label)

        return [self.tagSet[index] for index in sequence[::-1][1:]]


if __name__ == "__main__":
    crf = CRF(os.path.join(os.path.dirname( __file__ ),"..","data","partial","train"))
    pprint(crf.score_dict)
