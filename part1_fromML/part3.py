import numpy as np
def zerolistmaker(n):
    listofzeros = [-1e99] * n
    return listofzeros

def calculate_emission_parameter(train_data, train_x, train_y,tag_list,word_list):
    iter_num = 0  # can be deleted later
    print('----------------------------- Estimating the emission parameter ------------------------')
    emission_parameter = {}
    for word in word_list:
        emission_parameter[word] = [tag_list, zerolistmaker(len(tag_list))]  # the 1st sub_list stores tags, the 2nd sub_list stores score
    print('----------------------------- halfway here ------------------------')
    tag_dict = {}
    for idx, tag in enumerate(tag_list):
        tag_dict[tag]=idx
    pair_list = []
    for pair in train_data:
        if pair:
            if pair not in pair_list:
                pair_list.append(pair)
                T_id = tag_dict[pair[1]]
                emission_parameter[pair[0]][1][T_id]=float(np.log(train_data.count(pair) / train_y.count(pair[1])))
        # iter_num += 1
    print('----------------------------- Emission parameter is calculated -------------------------')
    print(emission_parameter)
    return emission_parameter


def calculate_transition_parameter(train_y,tag_list,tag_dict):
    iter_num = 0
    print('----------------------------- Estimating the transition parameter ------------------------')
    transition_parameter = {}
    train_y_set = []
    for i in tag_list:
        transition_parameter[i] = [tag_list,zerolistmaker(len(tag_list))]
    print('--------------------- 1/3 way --------------------------')
    for i in range(len(train_y)-1):
        train_y_set = train_y_set+[[train_y[i],train_y[i+1]]]
    print('--------------------- 2/3 way --------------------------')
    tagpair_list=[]
    for i in range(len(train_y)-1):
        if [train_y[i],train_y[i+1]] not in tagpair_list:
            tagpair_list.append([train_y[i],train_y[i+1]])
            T_id = tag_dict[train_y[i+1]]
            transition_parameter[train_y[i]][1][T_id]=float(np.log(train_y_set.count([train_y[i],train_y[i+1]]) / train_y.count(train_y[i])))
    print('----------------------------- Transition parameter is calculated -------------------------')
    print(transition_parameter)
    return transition_parameter


if __name__ == '__main__':

    path_train = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\train"
    path_in = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\dev.in"
    path_out = "C:\\Users\\87173\\Desktop\\term8\\NLP\\Coding HW\\NLP_project\\data\\partial\\dev.out"
    # train_lines = open(path_en_train).read().splitlines()

    train_lines = list(filter(None, open(path_train).read().splitlines()))

    train_data = [line.split() for line in train_lines]
    train_x = [line[0] for line in train_data if line]
    train_y = [line[1] for line in train_data if line]

    # train_data_em, train_x, train_y_em = modify_train_data_em(train_lines, train_path)
    # train_y_tr = modify_train_data_tr(train_lines, train_path_tr)
    tag_list_tr = list(dict.fromkeys(train_y))
    tag_list_em = list(dict.fromkeys(train_y))
    word_list = list(dict.fromkeys(train_x))
    tag_dict = {}
    for idx, tag in enumerate(tag_list_tr):
        tag_dict[tag]=idx
        print(idx)
    print(len(tag_list_tr))
    print(len(tag_list_em))
    emission_para = calculate_emission_parameter(train_data, train_x, train_y, tag_list_em, word_list)
    transition_para = calculate_transition_parameter(train_y, tag_list_tr, tag_dict)
