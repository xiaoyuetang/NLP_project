from collections import defaultdict
import tqdm
from pprint import pprint
import numpy as np
import os

from part1 import Feature, take

import warnings
warnings.filterwarnings("ignore")


class posFeature(Feature):
    def __init__(self, path_train_full, path_train_partial):
        self.path_train = path_train_partial
        self.path_train_full = path_train_full
        self.poses, self.label_pos = self.data_processing_pos()

        Feature.__init__(self, path_train_partial)
        self.words = dict(list(self.words.items()) + list(self.poses.items()))
        self.label_words = self.relate_labels_pos(self.label_words, self.label_pos)
        self.emission_parameter= self.calculate_emiss_parameter()
        self.transition_parameter = self.calculate_trans_parameter()
        self.feature_dict = self.calculate_feature()

    def data_processing_pos(self):
        poses = {}
        label_pos = {}

        for line in open(self.path_train_full, encoding='utf-8', mode='r'):
            segmented_line = line.strip()
            if segmented_line:
                pos, tag = self.line_cut_pos(segmented_line)
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
        return pos, tag

    def relate_labels_pos(self, label_words, label_pos):
        out_labels_words_pos = label_words.copy()
        for tag in label_words:
            out_dict = dict(list(label_words[tag].items()) + list(label_pos[tag].items()))
            out_labels_words_pos[tag] = out_dict
        return out_labels_words_pos


if __name__ == "__main__":
    dataset_full = os.path.join(os.path.dirname( __file__ ),"..", "data", "full")
    dataset_partial = os.path.join(os.path.dirname( __file__ ),"..", "data", "partial")
    pos_feature = posFeature(os.path.join(dataset_full, "train"), os.path.join(dataset_partial, "train"))
    feature_dict = pos_feature.feature_dict
    n_items = take(5, feature_dict.items())
    pprint(n_items)
    print("Number of features: ", len(feature_dict.items()))
