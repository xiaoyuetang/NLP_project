from collections import defaultdict
import numpy as np

def calculate_emission_parameter(train_data, train_x, train_y):
    # iter_num = 0  # can be deleted later
    print('----------------------------- Estimating the emission parameter ------------------------')
    emission_parameter = {}
    for word in train_x:
        emission_parameter[word] = [[], []]  # the 1st sub_list stores tags, the 2nd sub_list stores score

    for pair in train_data:
        if pair[1] not in emission_parameter[pair[0]][0]:
            emission_parameter[pair[0]][0].append(pair[1])
            emission_parameter[pair[0]][1].append(train_data.count(pair) / train_y.count(pair[1]))
        # iter_num += 1
    print('----------------------------- Emission parameter is calculated -------------------------')
    return emission_parameter


def calculate_feature(train_data, emission_parameter, transition_parameter):
    feature_dic = defaultdict()

    for pair in train_data:
        idx = emission_parameter[pair[0]][0].index(pair[1])
        feature_dic["emission:"+pair[1]+'+'+pair[0]] = np.log2(emission_parameter[pair[0]][1][idx])

    return feature_dic


if __name__ == '__main__':
    path_train = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\train"
    path_in = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\dev.in"
    path_out = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\dev.out"
    # train_lines = open(path_en_train).read().splitlines()

    train_lines = list(filter(None, open(path_train).read().splitlines()))

    train_data = [line.split() for line in train_lines]
    train_x = [line[0] for line in train_data if line]
    train_y = [line[1] for line in train_data if line]

    emission_para = calculate_emission_parameter(train_data, train_x, train_y)
    feature = calculate_feature(train_data, emission_para)
    print(feature)


