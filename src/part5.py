from collections import defaultdict
import tqdm
from pprint import pprint
import numpy as np
import os

from part1 import Feature, take
from part2 import CRF

import warnings
warnings.filterwarnings("ignore")


class posFeature(Feature):
    def __init__(self, path_train_full, path_train_partial):
        self.path_train = path_train_partial
        self.path_train_full = path_train_full
        self.poses, self.label_pos = self.data_processing_pos()

        Feature.__init__(self, path_train_partial)
        self.combined_parameter = {}
        self.words = dict(list(self.words.items()) + list(self.poses.items()))
        self.label_words = self.relate_labels_pos(self.label_words, self.label_pos)
        self.emission_parameter= self.calculate_emiss_parameter()
        self.transition_parameter = self.calculate_trans_parameter()
        self.feature_dict = self.calculate_feature()
        self.feature_dict_combined = self.calculate_combined_feature()

    def data_processing_pos(self):
        poses = {}
        label_pos = {}

        for line in open(self.path_train_full, encoding='utf-8', mode='r'):
            segmented_line = line.strip()
            if segmented_line:
                word, pos, tag = self.line_cut_pos(segmented_line)
                if pos not in poses:
                    poses[pos] = 1
                else:
                    poses[pos] += 1

                if tag not in label_pos:
                    label_pos[tag] = {pos: 1}
                else:
                    if pos not in label_pos[tag]:
                        label_pos[tag][pos] = 1
                    else:
                        label_pos[tag][pos] += 1

        return poses, label_pos

    def line_cut_pos(self, segmented_line):
        segmented_line = segmented_line.split(' ')
        word = segmented_line[0]
        pos = segmented_line[1]
        tag = segmented_line[2]
        return word, pos, tag

    def relate_labels_pos(self, label_words, label_pos):
        out_labels_words_pos = label_words.copy()
        for tag in label_words:
            out_dict = dict(list(label_words[tag].items()) + list(label_pos[tag].items()))
            out_labels_words_pos[tag] = out_dict
        return out_labels_words_pos

    def calculate_combined_feature(self):
        '''
        create feature dictionary by combining emission and transition probability.
        p(yi|xi,yi-1) = count(yi-1, yi, xi)/count(yi-1, xi)
        '''
        preT_X = {}
        count_feature = {}
        feature_dic = {}
        _preT = ''
        _newT = 'START'
        for line in open(self.path_train_full, encoding='utf-8', mode='r'):
            _preT = _newT if (_newT != 'STOP') else 'START'
            segmented_line = line.rstrip()

            if segmented_line:
                word, pos, tag = self.line_cut_pos(segmented_line)
                _newT = tag
            else:
                _newT = 'STOP'

            if (_preT, word) not in preT_X:
                preT_X[(_preT, word)] = 1
                count_feature[(_preT, word)] = {_newT: 1}
            else:
                preT_X[(_preT, word)] += 1
                if _newT not in count_feature[(_preT, word)]:
                    count_feature[(_preT, word)][_newT] = 1
                else:
                    count_feature[(_preT, word)][_newT] += 1

        self.combined_parameter = count_feature.copy()
        for pretag_word in count_feature:
            pretag, word = pretag_word[0], pretag_word[1]
            for tag in count_feature[pretag_word]:
                feature_dic["combine:" + pretag + "+" + tag + "+" + word] = np.log(count_feature[pretag_word][tag] / preT_X[pretag_word])
                self.combined_parameter[pretag_word][tag] = count_feature[pretag_word][tag] / preT_X[pretag_word]

        return feature_dic


class posCRF(CRF):
    def __init__(self, path_train_full, path_train_partial):
        self.path_train = path_train_partial
        self.path_train_full = path_train_full

        CRF.__init__(self, path_train_partial)
        self.feature = posFeature(path_train_full, path_train_partial)
        self.feature_dict = self.feature.feature_dict
        self.emission_parameter = self.feature.emission_parameter
        self.transition_parameter = self.feature.transition_parameter
        self.combined_parameter = self.feature.combined_parameter
        self.train_set = list(self.feature.words)

    def inference_pos(self, input_path, output_path, combine=False):
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
                prediction_sequence = self.viterbi_combine(sequence) if combine else self.viterbi_pos(sequence)
                for i in range(len(sequence)):
                    if prediction_sequence:
                        f.write('{0} {1}\n'.format(sequence[i], prediction_sequence[i]))

                    else:
                        f.write('{0} O\n' .format(sequence[i]))

                f.write('\n')
                sequence = []

        print ('Finished writing to file')
        return f.close()

    def viterbi_pos(self, sequence):
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

                emission = self.get_estimate(sequence, label, k)
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

    def viterbi_combine(self, sequence):
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

            try:
                pi[0][label] = [self.transition_parameter['START'][label] * emission \
                    * self.combined_parameter[('START', sequence[0].split(" ")[0])][label]]
            except:
                pi[0][label] = [self.transition_parameter['START'][label] * emission * 0.1e-8]

        for k in tqdm.tqdm(range(1, len(sequence))):
            for label in tags:
                piList=[]
                for transTag in tags:
                    if label not in self.transition_parameter[transTag]: continue
                    try:
                        score = pi[k-1][transTag][0] * self.transition_parameter[transTag][label] \
                            * self.combined_parameter[(transTag, sequence[k].split(" ")[0])][label]
                    except:
                        score = pi[k-1][transTag][0] * self.transition_parameter[transTag][label] * 0.1e-8



                    piList.append([score, transTag])
                piList.sort(reverse=True)
                pi[k][label]=piList[0]

                emission = self.get_estimate(sequence, label, k)
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

    def get_estimate(self, sequence, label, k):
        '''
        function to deal with unseen data.
        '''
        if sequence[k].split(" ")[0] in self.train_set:
            if sequence[k].split(" ")[0] in self.emission_parameter[label]:
                emission_word = self.emission_parameter[label][sequence[k].split(" ")[0]]
            else:
                emission_word  = 0.1e-8
        else:
            # emission_word = self.emission_parameter[label]['#UNK#']
            emission_word  = 0.1e-8

        if sequence[k].split(" ")[1] in self.train_set:
            if sequence[k].split(" ")[1] in self.emission_parameter[label]:
                emission_pos = self.emission_parameter[label][sequence[k].split(" ")[1]]
            else:
                emission_pos = 0.1e-8
        else:
            # emission_pos = self.emission_parameter[label]['#UNK#']
            emission_pos = 0.1e-8

        emission = emission_pos*emission_word

        return emission


class StructuredPerceptron():
    def __init__(self):
        self.objective_function = None


if __name__ == "__main__":
    dataset_full = os.path.join(os.path.dirname( __file__ ),"..", "data", "full")
    dataset_partial = os.path.join(os.path.dirname( __file__ ),"..", "data", "partial")
    pos_feature = posFeature(os.path.join(dataset_full, "train"), os.path.join(dataset_partial, "train"))
    feature_dict_combined = pos_feature.feature_dict_combined
    feature_dict = pos_feature.feature_dict
    n_items = take(5, feature_dict.items())
    pprint(n_items)
    print("Number of features: ", len(feature_dict.items()))

    pos_crf = posCRF(os.path.join(dataset_full, "train"), os.path.join(dataset_partial, "train"))
    input_path = os.path.join(dataset_full, "dev.in")
    output_path = os.path.join(dataset_full, "dev.p5.CRF.f3.out")
    pos_crf.inference_pos(input_path, output_path)

    input_path_combine = os.path.join(dataset_full, "dev.in")
    output_path_combine = os.path.join(dataset_full, "dev.p5.CRF.f4.out")
    pos_crf.inference_pos(input_path_combine, output_path_combine, combine=True)
