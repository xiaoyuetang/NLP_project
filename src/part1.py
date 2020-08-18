from collections import defaultdict
from itertools import islice
from pprint import pprint
import numpy as np
import os


def calculate_emission_parameter(train_data, train_x, train_y):
    # iter_num = 0  # can be deleted later
    emission_parameter = {}
    for word in train_x:
        emission_parameter[word] = [[], []]
        # the 1st sub_list stores tags, the 2nd sub_list stores score

    for pair in train_data:
        if pair[1] not in emission_parameter[pair[0]][0]:
            emission_parameter[pair[0]][0].append(pair[1])
            emission_parameter[pair[0]][1].append(train_data.count(pair) / train_y.count(pair[1]))
        # iter_num += 1
    return emission_parameter


def calculate_feature(train_data, emission_parameter):
    feature_dic = defaultdict()

    for pair in train_data:
        idx = emission_parameter[pair[0]][0].index(pair[1])
        feature_dic["emission:"+pair[1]+'+'+pair[0]] = np.log2(emission_parameter[pair[0]][1][idx])

    return feature_dic


def generate_data(path_train):
    train_lines = list(filter(None, open(path_train).read().splitlines()))

    train_data = [line.split() for line in train_lines]
    train_x = [line[0] for line in train_data if line]
    train_y = [line[1] for line in train_data if line]

    return train_data, train_x, train_y


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


path_train = os.path.join(os.path.dirname( __file__ ),"..","data","partial","train")
train_data, train_x, train_y = generate_data(path_train)
emission_para = calculate_emission_parameter(train_data, train_x, train_y)
feature_dict = calculate_feature(train_data, emission_para)


if __name__ == '__main__':
    n_items = take(5, feature_dict.items())
    pprint(n_items)
