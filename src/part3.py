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
        self.tags, self.words, self.label_words = self.feature.data_processing()
        self.crf = CRF(train_path)
        self.sentence = None

    def forward_algo(self, sentence, feature_dic, tags):
        # tags = self.feature.tags.keys()
        n = len(sentence)  # Number of words
        d = len(tags)  # Number of states
        forward_dic = np.zeros((n, d))
        score = 0
        for i, tag in enumerate(tags):
            transition_score = self.feature.get_feature_weight("START", tag, "transition")

        for i, tag in enumerate(tags):
            forward_dic[i] = {}
            forward_dic[i][0] = self.feature.get_feature_weight("START", tag, "transition") + \
                                self.feature.get_feature_weight(tag, sentence[0], "emission")
            # print(forward_dic[0][0])

        for j, word in enumerate(sentence[1:]):
            j += 1
            for i, tag in enumerate(tags):
                for k, tag_ in enumerate(tags):
                    total_sum = sum([np.exp(self.feature.get_feature_weight(tag_, tag, "transition") +
                                            self.feature.get_feature_weight(tag, word, "emission") +
                                            forward_dic[k][j-1])])
                    forward_dic[i][j] = np.log(total_sum)

        for i, tag in enumerate(tags):
            score += np.exp(forward_dic[i][len(sentence)-1] + self.feature.get_feature_weight(tag, "END", "transition"))
        score = np.log(score)

        return score, forward_dic

    def loss_function(self, sentences, tags, feature_dict, lr=0.1, reg=False):
        out = 0
        # term1 = self.crf.get_score()
        # term2_sum = 0
        for i, sentence in enumerate(sentences):
            labels = tags[i]
            term1 = self.crf.get_score(sentence, labels)
            term2, _ = self.forward_algo(sentence)
            out += term1-term2
            # print(term2)
            # term2_sum += term2
        if reg:
            reg_loss = 0
            for feature in feature_dict:
                reg_loss += feature_dict[feature] ** 2
            out += lr * reg_loss

        return -out

    def backward_algo(self, sentence):
        tags = self.feature.tags.keys()
        backward_dic = defaultdict()
        score = 0
        for i, tag in enumerate(tags):
            backward_dic[i] = {}
            backward_dic[i][len(sentence) - 1] = self.feature.get_feature_weight(tag, "END", "transition")

        for j in range(len(sentence)-1, 0, -1):
            word = sentence[j]
            for i, tag in enumerate(tags):
                total = sum([np.exp(self.feature.get_feature_weight(tag, tag_, 'transition') +
                                    self.feature.get_feature_weight(tag_, word, 'emission') +
                                    backward_dic[k][j]) for k, tag_ in enumerate(tags)])
                backward_dic[i][j-1] = np.log(total)

        for i, tag in enumerate(tags):
            score += np.exp(self.feature.get_feature_weight("START", tag, "transition") +
                            self.feature.get_feature_weight(tag, sentence[0], "emission") +
                            backward_dic[i][0])

        out = np.log(score)
        return out, backward_dic

    def calculate_expected_counts(self, sentence):
        tags = self.feature.tags.keys()

        f_score, f_dic = self.forward_algo(sentence)
        b_score, b_dic = self.backward_algo(sentence)

        expected_counts = defaultdict(float)

        for j, word in enumerate(sentence):
            for i, tag in enumerate(tags):
                value = f_dic[i][j] + b_dic[i][j] - f_score
                expected_counts["emission:{}+{}".format(tag, word)] += np.exp(value)

        for i, tag in enumerate(tags):
            start_val = f_dic[i][0] + b_dic[i][0] - f_score
            stop_val = f_dic[i][len(sentence)-1] + b_dic[i][len(sentence)-1] - f_score
            expected_counts["transition:{}+{}".format("START", tag)] += np.exp(start_val)
            expected_counts["transition:{}+{}".format(tag, "STOP")] += np.exp(stop_val)

        for i, tag in enumerate(tags):
            for j, tag_1 in enumerate(tags):
                total = 0
                transition = self.feature.get_feature_weight(tag, tag_1, "transition")
                for k, word in enumerate(sentence[1:]):
                    emission = self.feature.get_feature_weight(tag_1, word, "emission")
                    value = f_dic[i][k] + b_dic[j][k + 1] + transition + emission - f_score
                    total += np.exp(value)

                expected_counts["transition:{}+{}".format(tag, tag_1)] = total
        return expected_counts

    def calculate_counts(self, x, y):
        counts = defaultdict(int)
        for i, word in enumerate(x):
            tag = y[i]
            counts["emission:{}+{}".format(tag, word)] += 1

        y = ["START"] + y + ["STOP"]
        for i, tag in enumerate(y[:-1]):
            tag_1 = y[i + 1]
            counts["transition:{}+{}".format(tag, tag_1)] += 1
        return counts

    def calculate_gradients(self, x, y, feature_dict, lr=0.1, reg=False):
        gradient_dic = defaultdict(float)

        for i, sentence in enumerate(x):
            tags = y[i]

            expected_counts = self.calculate_expected_counts(sentence)
            pair_counts = self.calculate_counts(sentence, tags)

            for key, value in expected_counts.items():
                gradient_dic[key] += value

            for key, value in pair_counts.items():
                gradient_dic[key] -= value
        if reg:
            for key, value in feature_dict.items():
                gradient_dic[key] += 2 * lr * feature_dict[key]
            # print(feature_dict)
        # return None
        return gradient_dic


    def get_sentences_tags(self, train_path):
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
    sentence_list, tag_list = train.get_sentences_tags(train_path)

    print(len(sentence_list), len(tag_list))
    out = train.loss_function(sentence_list, tag_list, train.feature_dict)
    print(out)
    gradient = train.calculate_gradients(sentence_list, tag_list, train.feature_dict)
    print(gradient)