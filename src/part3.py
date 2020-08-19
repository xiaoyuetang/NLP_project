import os
import numpy as np
from collections import defaultdict

from part1 import Feature
from part2 import CRF

import warnings
warnings.filterwarnings("ignore")


class Train():
    def __init__(self, train_path):
        self.feature = Feature(train_path)
        self.feature_dict = self.feature.feature_dict
        self.crf = CRF(train_path)
        self.sentence = None

    def forward_algo(self, sentence):
        tags = self.feature.tags.keys()
        forward_dic = defaultdict()
        score = 0
        for i, tag in enumerate(tags):
            forward_dic[i] = {}
            forward_dic[i][0] = self.feature.get_feature_weight("START", tag, "transition") + \
                                self.feature.get_feature_weight(tag, sentence[0], "emission")
            # print(forward_dic[0][0])

        for j, word in enumerate(sentence[1:]):
            j += 1
            for i, tag in enumerate(tags):
                total_sum = sum([np.exp(self.feature.get_feature_weight(tag_, tag, "transition") +
                                        self.feature.get_feature_weight(tag, word, "emission") +
                                        forward_dic[k][j-1]) for k, tag_ in enumerate(tags)])
                forward_dic[i][j] = np.log(total_sum)

        for i, tag in enumerate(tags):
            score += np.exp(forward_dic[i][len(sentence)-1] + self.feature.get_feature_weight(tag, "END", "transition"))
        score = np.log(score)

        return score

    def loss_function(self, sentences, tags):
        out = 0
        # term1 = self.crf.get_score()
        # term2_sum = 0
        for i, sentence in enumerate(sentences):
            labels = tags[i]
            term1 = self.crf.get_score(sentence, labels)
            term2 = self.forward_algo(sentence)
            out = term1-term2
            # print(term2)
            # term2_sum += term2
        return -out


def get_sentences_tags(train_path):
    with open(train_path) as file:
        sentence_list = []
        words = []
        tag_list = []
        tags = []
        for line in file:
            if line == "\n":
                sentence_list.append(words)
                tag_list.append(tags)
                words = []
                tags = []
            else:
                word, tag = line.split()
                words.append(word.strip())
                tags.append(tag.strip())
    return sentence_list, tag_list



if __name__ == '__main__':
    dataset = os.path.join(os.path.dirname(__file__), "..", "data", "partial")
    train_path = os.path.join(dataset, "train")
    train = Train(train_path)

    sentences, tags = get_sentences_tags(train_path)
    out = train.loss_function(sentences, tags)
    print(out)
