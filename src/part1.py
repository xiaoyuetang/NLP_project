from collections import defaultdict
import numpy as np


def zerolistmaker(n):
    listofzeros = [-1e99] * n
    return listofzeros


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


def estimateTransition(filePath):
    tags = {}
    transitionTag = {}
    transitionProbability = {}
    _preT = ''
    _newT = 'START'
    for line in open(filePath, encoding='utf-8', mode='r'):
        _preT = _newT if (_newT != 'STOP') else 'START'
        segmentedLine = line.rstrip()

        if segmentedLine:
            segmentedLine = segmentedLine.rsplit(' ', 1)
            _newT = segmentedLine[1]
        else:
            _newT = 'STOP'
        if _preT not in tags:
            tags[_preT] = 1
            transitionTag[_preT] = {_newT: 1}
        else:
            tags[_preT] += 1
            if _newT not in transitionTag[_preT]:
                transitionTag[_preT][_newT] = 1
            else:
                transitionTag[_preT][_newT] += 1

    calculateTransitionProbability(transitionProbability, transitionTag, tags)

    return transitionProbability


def calculateTransitionProbability(transitionProbability, transitionTag, tags):
    for tag in transitionTag:
        transitionProbability[tag] = {}
        for transition in transitionTag[tag]:
            transitionProbability[tag][transition] = transitionTag[tag][transition] / tags[tag]



def calculate_feature(train_data, emission_parameter, transition_parameter):
    feature_dic = defaultdict()

    # for pair in train_data:
    #     idx = emission_parameter[pair[0]][0].index(pair[1])
    #     feature_dic["emission:"+pair[1]+'+'+pair[0]] = np.log2(emission_parameter[pair[0]][1][idx])

    for word in emission_parameter.keys():
        for idx, tag in enumerate(emission_parameter[word][0]):
            feature_dic["emission:" + tag + '+' + word] = np.log2(emission_parameter[word][1][idx])
    for word in transition_parameter.keys():
        for word2 in transition_parameter[word]:
            feature_dic["transition:" + word + '+' + word2] = np.log2(transition_parameter[word][word2])


    return feature_dic


if __name__ == '__main__':
    path_train = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\train"
    path_in = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\dev.in"
    path_out = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\dev.out"
    modified_train_path_tr = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\train_modify"
    # train_lines = open(path_en_train).read().splitlines()

    train_lines = list(filter(None, open(path_train).read().splitlines()))
    train_data = [line.split() for line in train_lines]
    train_x = [line[0] for line in train_data if line]
    train_y = [line[1] for line in train_data if line]

    tag_list_tr = list(dict.fromkeys(train_y))
    tag_dict = {}
    for idx, tag in enumerate(train_y):
        tag_dict[tag] = idx

    emission_para = calculate_emission_parameter(train_data, train_x, train_y)

    # train_y_tr = modify_train_data_tr(train_lines, modified_train_path_tr)
    # tag_list_tr = list(dict.fromkeys(train_y_tr))
    # tag_dict = {}
    # for idx, tag in enumerate(tag_list_tr):
    #     tag_dict[tag]=idx
    # transition_para = calculate_transition_parameter(train_y_tr, tag_list_tr, tag_dict)
    # # print(transition_para)
    transition_para = estimateTransition(path_train)

    feature = calculate_feature(train_data, emission_para, transition_para)
    print(feature)


