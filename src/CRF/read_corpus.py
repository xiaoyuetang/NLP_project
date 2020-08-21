#Function used to read training file
def read_train_file(filename):
    data = list()
    segmentLine = list(open(filename,encoding='UTF-8',mode='r'))
    word = list()
    tag = list()
    for data_string in segmentLine:
        words = data_string.strip().split()
        if len(words) == 0:
            #seperate sentenses
            data.append((word, tag))
            word = list()
            tag = list()
        else:
            word.append(words[:-1])
            tag.append(words[-1])
    if len(word) > 0:
        data.append((word, tag))
    return data

#Function used to read test file
def read_test_file(filename):
    data = list()
    segmentLine = list(open(filename,encoding='utf-8',mode='r'))
    word = list()
    for data_string in segmentLine:
        words = data_string.strip().split()
        if len(words) == 0:
            data.append((word))
            word = list()
        else:
            word.append(words)
    if len(word) > 0:
        data.append((word))

    return data
