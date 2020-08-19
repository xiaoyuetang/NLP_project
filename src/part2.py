from collections import defaultdict
from pprint import pprint
import numpy as np
import os

from part1 import feature_dict, train_x, train_y


class CRF():
    def __init__(self):
        self.training_data = None
        self.feature_set = None
        self.tagSet = None
        self.tagArray = None
        self.labelCounts = None
        self.params = None

    def getFeature(self):
        return [[self.feature_set.get_feature_list(word, t) for t in range(len(word))]
                for word, _ in self.training_data]

    def estimation(self):
        training_feature_data = self.getFeature()
        print('Start L-BFGS-B')
        # Minimize a function func using the L-BFGS-B algorithm.
        # magic
        self.params, log_likelihood, information = fmin_l_bfgs_b(
                                    func=_log_likelihood,
                                    fprime=_gradient,x0=np.zeros(len(self.feature_set)),
                                    args=(self.training_data,
                                    self.feature_set,
                                    training_feature_data,
                                    self.feature_set.get_empirical_counts(),
                                    self.tagSet,
                                    VARIANCE),
                                    callback=_callback)
        print('Finish!')

    #Main function to train model
    def train(self, inputFilename, model_filename):
        # Read the training corpus
        print("Reading data")
        self.training_data = readTrainFile(inputFilename)
        print("Reading Finish!")

        # Generate feature set from the input files
        self.feature_set = FeatureSet()
        self.feature_set.scan(self.training_data)
        self.tagSet, self.tagArray = self.feature_set.get_labels()
        self.labelCounts = len(self.tagArray)
        print("Number of labels:",self.labelCounts-1)
        print("Number of features: ",len(self.feature_set))

        # Estimates params to maximize log-likelihood of the input file
        self.estimation()
        self.save_model(model_filename)

        print('Training Finished' )

    #Main function to test the input file
    def test(self, test_inputFilename):
        test_data = readTestFile(test_inputFilename)
        with open ('dev.p5.out',encoding='UTF-8',mode='w') as f:
            for word in test_data:
                layer=len(word)
                Yprime = self.inference(word,layer)
                #Write the predicted result to the file
                for t in range(len(word)):
                    f.write(word[t][0])
                    f.write(' ')
                    f.write(Yprime[t])
                    f.write('\n')
                f.write('\n')

    def save_model(self, model_filename):
        #model saved as CRF++ format
        model = {"feature_dic": self.feature_set.serialize_feature_dic(),
                 "num_features": self.feature_set.num_features,
                 "labels": self.feature_set.tagArray,
                 "params": list(self.params)}
        f = open(model_filename,encoding='utf-8',mode= 'w')
        json.dump(model, f, ensure_ascii=False, indent=2, separators=(',', ':'))
        f.close()
        print('Trained Model saved')

    def inference(self, word,layer):
        potential_table = generate_potential_table(self.params, self.labelCounts,self.feature_set, word, inference=True)
        #Use viterbi algorithm to decode the possible sequences
        Yprime = self.viterbi(word, potential_table,layer)
        return Yprime

    def viterbi(self, word, potential_table,layer):
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

    def save_model(self, model_filename):
        #model saved as CRF++ format
        model = {"feature_dic": self.feature_set.serialize_feature_dic(),
                 "num_features": self.feature_set.num_features,
                 "labels": self.feature_set.tagArray,
                 "params": list(self.params)}
        f = open(model_filename,encoding='utf-8',mode= 'w')
        json.dump(model, f, ensure_ascii=False, indent=2, separators=(',', ':'))
        f.close()
        print('Trained CRF Model has been saved')

    def load(self, model_filename):
        f = open(model_filename,encoding='utf-8',mode='r')
        model = json.load(f)
        f.close()
        self.feature_set = FeatureSet()
        self.feature_set.load(model['feature_dic'], model['num_features'], model['labels'])
        self.tagSet, self.tagArray = self.feature_set.get_labels()
        self.labelCounts = len(self.tagArray)
        self.params = np.array(model['params'])

        print('CRF model loaded')


    def feature_function(sequence_pair):
        '''
        a function that returns the number of times that the j-th feature in the
        feature_dictionary appears in sequence pair (x_seq, y_seq).
        '''
        count_dict = dict()
        for feature, weight in feature_dict.items():
            feature_list = feature.split('+')
            if "emission:" in feature_list[0]:
                feature_tuple = (feature_list[1], feature_list[0].replace("emission:",''))
                if feature_tuple in sequence_pair:
                    if feature in count_dict:
                        count_dict[feature] += 1
                    else:
                        count_dict[feature] = 1

        return count_dict


    def calculate_score(sequence_pair):
        '''
        a function to calculate the score for a given pair of input and output
        sequence pair (x_seq, y_seq), based on the above-mentioned features
        and weights used in Part 1.

        '''
        score_dict = dict()
        count_dict = feature_function(sequence_pair)
        for feature, weight in feature_dict.items():
            if feature in count_dict:
                score_dict[feature] = weight*count_dict[feature]

        return score_dict


def generate_seqpairs():
    sequence_pair = []
    for x, y in zip(train_x, train_y):
        sequence_pair.append((x,y))

    return sequence_pair


if __name__ == "__main__":
    sequence_pair = generate_seqpairs()
    pprint(calculate_score(sequence_pair))
