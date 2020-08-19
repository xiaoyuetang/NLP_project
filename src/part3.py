import os
import numpy as np
from collections import defaultdict

from part1 import Feature
from part2 import CRF


class Train():
    def __init__(self, train_path):
        self.feature = Feature(train_path)
        self.feature_dict = self.feature.feature_dict
        self.crf = CRF(train_path)
        self.sentence = None

    def forward_algo(self, sentence):
        tags = feature.tags.keys()
        forward_dic = defaultdict()
        score = 0
        for i, tag in enumerate(tags):
            forward_dic[i] = {}
            forward_dic[i][0] = feature.get_feature_weight("START", tag, "transition") + \
                                feature.get_feature_weight(tag, sentence[0], "emission")

        for j, word in enumerate(sentence[1:]):
            j += 1
            for i, tag in enumerate(tags):
                total_sum = sum([np.exp(feature.get_feature_weight(tag_, tag, "transition") +
                                        feature.get_feature_weight(tag, word, "emission") +
                                        forward_dic[k, j-1]) for k, tag_ in enumerate(tags)])
                forward_dic[i][j] = np.log2(total_sum)

        for i, tag in enumerate(tags):
            score += np.exp(forward_dic[i][-1] + feature.get_feature_weight(tag, "END", "transition"))
        score = np.log2(score)

        return score

    def loss_function(self, sentences, tags):
        out = 0
        for i, sentence in enumerate(sentences):
            labels = tags[i]
            # term1 = crf.get_score(sentence, labels, CRF)
            term1 = crf.get_score()
            _, term2 = forward_algo(sentence)
            # print(term2)
            out += term1 - term2
        out *= -1
        return out


if __name__ == '__main__':
    dataset = os.path.join(os.path.dirname( __file__ ),"..", "data", "partial")
    train = Train(os.path.join(dataset, "train"))
