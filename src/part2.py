from collections import defaultdict
from pprint import pprint
import numpy as np
import os

from part1 import feature_dict, train_x, train_y


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


def viterbi():
    '''
    using the Viterbi algorithm to find the most probable output
    sequence y∗ for an input sequence x.
    '''
    pass


def generate_seqpairs():
    sequence_pair = []
    for x, y in zip(train_x, train_y):
        sequence_pair.append((x,y))

    return sequence_pair


if __name__ == "__main__":
    sequence_pair = generate_seqpairs()
    pprint(calculate_score(sequence_pair))
