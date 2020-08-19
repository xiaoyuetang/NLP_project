import os
from conlleval import evaluate


def eval(pred_path, gold_path, out_path):
    f_pred = open(pred_path, encoding = 'utf-8')
    f_gold = open(gold_path, encoding = 'utf-8')
    data_pred = f_pred.readlines()
    data_gold = f_gold.readlines()
    gold_tags = list()
    pred_tags = list()

    with open(out_path, "w") as wf:
        for sentence in range(len(data_pred)):

            words_pred = data_pred[sentence].strip().split(' ')
            words_gold = data_gold[sentence].strip().split(' ')

            if len(words_gold)==1:
                continue

            gold_tags.append(words_gold[1])
            pred_tags.append(words_pred[1])
            wf.write(words_pred[0] + " " + words_pred[1] + " " + words_gold[1] +"\n")

    print(evaluate(gold_tags, pred_tags, verbose=True))


if __name__ == "__main__":
    dataset = os.path.join(os.path.dirname( __file__ ),"..", "data", "partial")
    eval(dataset+'/dev.p2.out', dataset+'/dev.out', dataset+'/eval.p2.out')
