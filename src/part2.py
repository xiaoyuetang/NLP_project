from collections import defaultdict
import tqdm
from pprint import pprint
import numpy as np
import os

from part1 import Feature, take
import warnings
warnings.filterwarnings("ignore")


class CRF():
    def __init__(self, path_train):
        self.feature = Feature(path_train)
        self.feature_dict = self.feature.feature_dict
        self.label_words = self.feature.label_words
        self.emission_parameter = self.feature.emission_parameter
        self.transition_parameter = self.feature.transition_parameter
        self.train_set = list(self.feature.words)

    def get_score(self, x, y):
        '''
        a function to calculate the score for a given pair of input and output
        sequence pair (x_seq, y_seq), based on the above-mentioned features
        and weights used in Part 1.
        '''
        dic = defaultdict(int)

        # count number of times each feature occured in the provided sequence
        for i in range(len(x)):
            dic["emission:{}+{}".format(y[i], x[i])] += 1
        y = ["START"] + y + ["STOP"]
        for i in range(1, len(y)):
            dic["transition:{}+{}".format(y[i - 1], y[i])] += 1

        # multiply weights with the number of occurences for each feature
        out = 0
        for key in dic:
            out += dic[key] * self.feature_dict[key]
        return out

    def inference(self, input_path, output_path):
        '''
        an inference for using Viterbi.
        '''
        print("Start Viterbi...")
        f = open(output_path, encoding='utf-8', mode= 'w')
        sequence = []
        for line in open(input_path, encoding='utf-8', mode='r'):
            word = line.rstrip()
            if word:
                sequence.append(word)
            else:
                prediction_sequence = self.viterbi(sequence)
                for i in range(len(sequence)):
                    if prediction_sequence:
                        f.write('{0} {1}\n'.format(sequence[i], prediction_sequence[i]))

                    else:
                        f.write('{0} O\n' .format(sequence[i]))

                f.write('\n')
                sequence = []

        print ('Finished writing to file')
        return f.close()

    def viterbi(self, sequence):
        '''
        perform decoding using the Viterbi algorithm to find the most
        probable output sequence y∗ for an input sequence x.
        '''
        tags = list(self.emission_parameter)

        pi = [{tag: [0.0, ''] for tag in tags} for o in sequence]

        # Initialization  stage
        for label in tags :
            if label not in self.transition_parameter['START']: continue
            emission = self.get_estimate(sequence, label)

            pi[0][label] = [self.transition_parameter['START'][label] * emission]

        for k in tqdm.tqdm(range(1, len(sequence))):
            for label in tags:
                piList=[]
                for transTag in tags:
                    if label not in self.transition_parameter[transTag]: continue
                    score = pi[k-1][transTag][0] * self.transition_parameter[transTag][label]
                    piList.append([score, transTag])
                piList.sort(reverse=True)
                pi[k][label]=piList[0]

                if sequence[k] in self.train_set:
                    if sequence[k] in self.emission_parameter[label]:
                        emission = self.emission_parameter[label][sequence[k]]
                    else:
                        emission = 0.1e-8
                else:
                    emission = self.emission_parameter[label]['#UNK#']
                pi[k][label][0] *= emission

        # Finally
        slist=[]
        result = [0.0, '']
        for trans_tag in tags:
            if 'STOP' not in self.transition_parameter[trans_tag]: continue
            score = pi[-1][trans_tag][0] * self.transition_parameter[trans_tag]['STOP']
            slist.append([score, trans_tag])
        slist.sort(reverse=True)
        result = slist[0]

        # Backtracking
        if not result[1]:
            return

        prediction = [result[1]]
        for k in reversed(range(len(sequence))):
            if k == 0: break
            prediction.insert(0, pi[k][prediction[0]][1])

        return prediction

    def get_estimate(self, sequence, label):
        '''
        function to deal with unseen data.
        '''
        k = 0
        if sequence[k] in self.train_set:
            if sequence[k] in self.emission_parameter[label]:
                emission = self.emission_parameter[label][sequence[k]]
            else:
                emission = 0.1e-8
        else:
            emission = self.emission_parameter[label]['#UNK#']
        return emission



if __name__ == "__main__":
    dataset = os.path.join(os.path.dirname( __file__ ),"..", "data", "partial")
    crf = CRF(os.path.join(dataset, "train"))
    input_path = os.path.join(dataset, "dev.in")
    output_path = os.path.join(dataset, "dev.p2.out")
    crf.inference(input_path, output_path)
