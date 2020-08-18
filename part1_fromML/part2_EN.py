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


def calculate_feature_1(train_data, emission_parameter):
    emission_parameter_dic = defaultdict()

    for pair in train_data:
        idx = emission_parameter[pair[0]][0].index(pair[1])
        emission_parameter_dic["emission:"+pair[1]+'+'+pair[0]] = np.log2(emission_parameter[pair[0]][1][idx])

    return emission_parameter_dic


if __name__ == '__main__':
    path_en_train = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\train"
    path_en_in = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\dev.in"
    path_en_out = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\dev.out"
    # train_lines = open(path_en_train).read().splitlines()

    train_lines = list(filter(None, open(path_en_train).read().splitlines()))
    # in_lines = list(filter(None, open(path_en_in).read().splitlines()))
    in_lines = open(path_en_in, encoding='utf-8').read().splitlines()

    train_data = [line.split() for line in train_lines]
    train_x = [line[0] for line in train_data if line]
    train_y = [line[1] for line in train_data if line]

    emission_para = calculate_emission_parameter(train_data, train_x, train_y)
    feature_1 = calculate_feature_1(train_data, emission_para)
    print(feature_1)


