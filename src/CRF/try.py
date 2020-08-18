from feature import FeatureSet
from crf import readTrainFile
from pprint import pprint
from itertools import islice


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


training_data = readTrainFile("data/partial/train")
feature_set = FeatureSet()
feature_set.scan(training_data)
tagSet, tagArray = feature_set.get_labels()
labelCounts = len(tagArray)
print("Number of labels:",labelCounts-1)
print("Number of features: ",len(feature_set))
n_items = take(5, feature_set.feature_dic.items())
pprint(n_items)
