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

            elif len(words_gold)==2:
                gold_tags.append(words_gold[1])
                pred_tags.append(words_pred[1])
                wf.write(words_pred[0] + " " + words_pred[1] + " " + words_gold[1] +"\n")

            else:
                gold_tags.append(words_gold[2])
                pred_tags.append(words_pred[2])
                wf.write(words_pred[0] + " " + words_pred[2] + " " + words_gold[2] +"\n")

    print(evaluate(gold_tags, pred_tags, verbose=True), "\n")


if __name__ == "__main__":
    dataset_full = os.path.join(os.path.dirname( __file__ ),"..", "data", "full")
    dataset_partial = os.path.join(os.path.dirname( __file__ ),"..", "data", "partial")
    print("PART II")
    eval(dataset_partial+'/dev.p2.out', dataset_partial+'/dev.out', dataset_partial+'/eval.p2.out')
    print("PART V (i)")
    eval(dataset_full+'/dev.p5.CRF.f3.out', dataset_full+'/dev.out', dataset_full+'/eval.p5.CRF.f3.out')
    print("PART V (ii)")
    eval(dataset_full+'/dev.p5.CRF.f4.out', dataset_full+'/dev.out', dataset_full+'/eval.p5.CRF.f4.out')
    print("PART V (iii)")
    eval(dataset_full+'/dev.p5.SP.out', dataset_full+'/dev.out', dataset_full+'/eval.p5.SP.out')
