from collections import defaultdict
import tqdm
from pprint import pprint
import numpy as np
import os

from part5 import posCRF, posFeature, take

import warnings
warnings.filterwarnings("ignore")


class myFeature(posFeature):
    def __init__(self, path_train_full, path_train_partial):
        posFeature.__init__(self, path_train_full, path_train_partial)
        self.my_feature_parameter1, self.my_feature_parameter2 = self.calculate_my_feature()

    def calculate_my_feature(self):
        def _feature1():
            '''
            create new feature:
            p(yi|xi, posi) = count(yi, xi, posi)/count(posi, xi)
            '''
            count_yi_xi_posi = {}
            count_posi_xi = {}
            my_feature_parameter = {}
            for line in open(self.path_train_full, encoding='utf-8', mode='r'):
                check = line.strip()
                if check:
                    word, pos, tag = line.split(" ")
                    if (word, pos, tag) not in count_yi_xi_posi:
                        count_yi_xi_posi[(word, pos, tag)] = 1
                    else:
                        count_yi_xi_posi[(word, pos, tag)] += 1

                    if (word, pos) not in count_posi_xi:
                        count_posi_xi[(word, pos)] = 1
                    else:
                        count_posi_xi[(word, pos)] += 1

            for threeple in count_yi_xi_posi:
                word, pos, tag = threeple
                my_feature_parameter[(word, pos)] = {}
                try:
                    my_feature_parameter[(word, pos)][tag] = count_yi_xi_posi[threeple] / count_posi_xi[(word, pos)]
                except:
                    my_feature_parameter[(word, pos)][tag] = 0.1e-8

            return my_feature_parameter

        def _feature2():
            '''
            create new feature:
            p(yi|yi-1, yi-2) = count(yi, yi-1, yi-2)/count(yi-1, yi-2)
            '''
            count_yi_yib1_yib2 = {}
            count_yib1_yib2 = {}
            my_feature_parameter = {}
            _preT = ''
            _newT = 'START'
            for line in open(self.path_train_full, encoding='utf-8', mode='r'):
                _prepreT = _preT
                _preT = _newT if (_newT != 'STOP') else 'START'
                check = line.rstrip()

                if check:
                    word, pos, tag = line.split(" ")
                    _newT = tag
                else:
                    _newT = 'STOP'

                if (_prepreT, _preT) not in count_yib1_yib2:
                    count_yib1_yib2[(_prepreT, _preT)] = 1
                    count_yi_yib1_yib2[(_prepreT, _preT)] = {_newT: 1}
                else:
                    count_yib1_yib2[(_prepreT, _preT)] += 1
                    if _newT not in count_yi_yib1_yib2[(_prepreT, _preT)]:
                        count_yi_yib1_yib2[(_prepreT, _preT)][_newT] = 1
                    else:
                        count_yi_yib1_yib2[(_prepreT, _preT)][_newT] += 1

            for yib1_yib2 in count_yi_yib1_yib2:
                my_feature_parameter[yib1_yib2] = {}
                for tag in count_yi_yib1_yib2[yib1_yib2]:
                    my_feature_parameter[yib1_yib2] = count_yi_yib1_yib2[yib1_yib2][tag] / count_yib1_yib2[yib1_yib2

                    ]

            return my_feature_parameter

        return _feature1(), _feature2()


class myCRF(posCRF):
    def __init__(self, path_train_full, path_train_partial):
        posCRF.__init__(self, path_train_full, path_train_partial)
        self.my_feature = myFeature(path_train_full, path_train_partial)
        self.my_feature_parameter1, self.my_feature_parameter2 = \
            self.my_feature.my_feature_parameter1, self.my_feature.my_feature_parameter2

    def inference_my(self, input_path, output_path):
        '''
        an inference for using my Viterbi.
        '''
        print("Start Viterbi...")
        f = open(output_path, encoding='utf-8', mode= 'w')
        sequence = []
        for line in open(input_path, encoding='utf-8', mode='r'):
            word = line.rstrip()
            if word:
                sequence.append(word)
            else:
                prediction_sequence = self.viterbi_my(sequence)
                for i in range(len(sequence)):
                    if prediction_sequence:
                        f.write('{0} {1}\n'.format(sequence[i], prediction_sequence[i]))

                    else:
                        f.write('{0} O\n' .format(sequence[i]))

                f.write('\n')
                sequence = []

        print ('Finished writing to file')
        return f.close()

    def viterbi_my(self, sequence):
        '''
        perform decoding using the Viterbi algorithm to find the most
        probable output sequence y∗ for an input sequence x.
        '''
        tags = list(self.emission_parameter)

        pi = [{tag: [0.0, ''] for tag in tags} for o in sequence]

        # Initialization  stage
        for label in tags :
            if label not in self.transition_parameter['START']: continue
            emission = self.get_estimate(sequence, label, 0)
            my_feature_score = self.get_my_feature1_score(sequence, label, 0)

            pi[0][label] = [self.transition_parameter['START'][label] * emission * my_feature_score]

        for k in tqdm.tqdm(range(1, len(sequence))):
            for label in tags:
                piList=[]
                for transTag in tags:
                    for prepreT in tags:
                        if label not in self.transition_parameter[transTag]: continue
                        # my_feature_score = self.get_my_feature2_score(prepreT, transTag, label)
                        score = pi[k-1][transTag][0] * self.transition_parameter[transTag][label] \
                            # * my_feature_score
                        piList.append([score, transTag])
                piList.sort(reverse=True)
                pi[k][label]=piList[0]

                emission = self.get_estimate(sequence, label, k)
                my_feature_score = self.get_my_feature1_score(sequence, label, k)
                pi[k][label][0] *= emission
                pi[k][label][0] *= my_feature_score

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

    def get_my_feature1_score(self, sequence, label, k):
        pair = sequence[k].split(" ")
        word, pos = pair[0], pair[1]
        if (word, pos) in self.my_feature_parameter1:
            if label in self.my_feature_parameter1[(word, pos)]:
                score = self.my_feature_parameter1[(word, pos)][label]
            else:
                score = 0.1e-8
        else:
            score = 0.1e-8

        return score

    def get_my_feature2_score(self, prepreT, preT, label):
        if (prepreT, preT) in self.my_feature_parameter2:
            if label in self.my_feature_parameter2[(prepreT, preT)]:
                score = self.my_feature_parameter2[(prepreT, preT)][label]
            else:
                score = 0.1e-8
        else:
            score = 0.1e-8

        return score


if __name__ == "__main__":
    dataset_full = os.path.join(os.path.dirname( __file__ ),"..", "data", "full")
    dataset_partial = os.path.join(os.path.dirname( __file__ ),"..", "data", "partial")

    test_full = os.path.join(os.path.dirname( __file__ ),"..", "test", "full")

    my_crf = myCRF(os.path.join(dataset_full, "train"), os.path.join(dataset_partial, "train"))
    input_path = os.path.join(dataset_full, "dev.in")
    output_path = os.path.join(dataset_full, "dev.p6.CRF.out")
    my_crf.inference_my(input_path, output_path)

    input_path_test = os.path.join(test_full, "test.in")
    output_path_test = os.path.join(test_full, "test.p6.CRF.out")
    my_crf.inference_my(input_path_test, output_path_test)
