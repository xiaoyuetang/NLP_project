from collections import defaultdict
from pprint import pprint
import numpy as np
import os

from part1 import Feature


class CRF():
    def __init__(self, path_train):
        self.feature = Feature(path_train)
        self.feature_dict = self.feature.feature_dict
        self.emission_para = self.feature.emission_para
        self.transition_para = self.feature.transition_para
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

    def inference(self, input_path, output_path):
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
        tags = list(self.emission_para)

        pi = [{tag: [0.0, ''] for tag in tags} for o in sequence]

        # Initialization  stage
        for label in tags :
            if label not in transitionEstimates['##START##']: continue
            emission=getEstimate(sequence,trainingSet,label)


            pi[0][label] = [transitionEstimates['##START##'][label] * emission]

        for k in tqdm.tqdm(range(1, len(sequence))):
            for label in tags:
                piList=[]
                for transTag in tags:
                    if label not in transitionEstimates[transTag]: continue
                    score = pi[k-1][transTag][0] * transitionEstimates[transTag][label]
                    piList.append([score, transTag])
                piList.sort(reverse=True)
                pi[k][label]=piList[0]

                if sequence[k] in trainingSet:
                    if sequence[k] in emissionEstimates[label]:
                        emission = emissionEstimates[label][sequence[k]]
                    else:
                        emission = 0.0
                else:
                    emission = emissionEstimates[label]['#UNK#']
                pi[k][label][0] *= emission

        # Finally
        slist=[]
        result = [0.0, '']
        for transTag in tags:
            if '##STOP##' not in transitionEstimates[transTag]: continue
            score = pi[-1][transTag][0] * transitionEstimates[transTag]['##STOP##']
            slist.append([score, transTag])
        slist.sort(reverse=True)
        result=slist[0]
        # Backtracking
        if not result[1]:
            return

        prediction = [result[1]]
        for k in reversed(range(len(sequence))):
            if k == 0: break
            prediction.insert(0, pi[k][prediction[0]][1])

        return prediction



if __name__ == "__main__":
    crf = CRF(os.path.join(os.path.dirname( __file__ ),"..","data","partial","train"))
    pprint(crf.score_dict)
