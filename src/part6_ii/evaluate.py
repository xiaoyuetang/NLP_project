import time
from collections import Counter
from os.path import join
# from models.hmm import HMM
# from models.crf import CRFModel
from models.bilstm_crf import BILSTM_Model
from utils import save_model, flatten_lists
# from evaluating import Metrics


# def hmm_train_eval(train_data, test_data, word2id, tag2id, remove_O=False):
#     """训练并评估hmm模型"""
#     # 训练HMM模型
#     train_word_lists, train_tag_lists = train_data
#     test_word_lists, test_tag_lists = test_data
#
#     hmm_model = HMM(len(tag2id), len(word2id))
#     hmm_model.train(train_word_lists,
#                     train_tag_lists,
#                     word2id,
#                     tag2id)
#     save_model(hmm_model, "./ckpts/hmm.pkl")
#
#     # 评估hmm模型
#     pred_tag_lists = hmm_model.test(test_word_lists,
#                                     word2id,
#                                     tag2id)
#
#     metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
#     metrics.report_scores()
#     metrics.report_confusion_matrix()
#
#     return pred_tag_lists


# def crf_train_eval(train_data, test_data, remove_O=False):
#
#     # 训练CRF模型
#     train_word_lists, train_tag_lists = train_data
#     test_word_lists, test_tag_lists = test_data
#
#     crf_model = CRFModel()
#     crf_model.train(train_word_lists, train_tag_lists)
#     save_model(crf_model, "./ckpts/crf.pkl")
#
#     pred_tag_lists = crf_model.test(test_word_lists)
#
#     metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
#     metrics.report_scores()
#     metrics.report_confusion_matrix()
#
#     return pred_tag_lists


def bilstm_train_and_eval(train_data, dev_data, test_data,
                          word2id, tag2id, crf=True, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)

    model_name = "bilstm_crf" if crf else "bilstm"
    save_model(bilstm_model, "./ckpts/"+model_name+".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)

    # metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    # metrics.report_scores()
    # metrics.report_confusion_matrix()
    data_dir = "../../data/partial"
    generate_output_file(open(join(data_dir, 'test.in')), open(join(data_dir, 'test.p6.model.out')), pred_tag_lists)
    print("*** ouput is saved.")
    return pred_tag_lists


def generate_output_file(input_path, output_path, prediction_sequence):
    print("Start Viterbi...")
    f = open(output_path, encoding='utf-8', mode='w')
    sequence = []
    for line in open(input_path, encoding='utf-8', mode='r'):
        word = line.rstrip()
        if word:
            sequence.append(word)
        else:
            # prediction_sequence = self.viterbi_my(sequence)
            for i in range(len(sequence)):
                if prediction_sequence:
                    f.write('{0} {1}\n'.format(sequence[i], prediction_sequence[i]))

                else:
                    f.write('{0} O\n'.format(sequence[i]))

            f.write('\n')
            sequence = []

    print('Finished writing to file')
    return f.close()


def ensemble_evaluate(results, targets, remove_O=False):
    """ensemble多个模型"""
    for i in range(len(results)):
        results[i] = flatten_lists(results[i])

    pred_tags = []
    for result in zip(*results):
        ensemble_tag = Counter(result).most_common(1)[0][0]
        pred_tags.append(ensemble_tag)

    targets = flatten_lists(targets)
    assert len(pred_tags) == len(targets)

    print("Ensemble 四个模型的结果如下：no")
    # metrics = Metrics(targets, pred_tags, remove_O=remove_O)
    # metrics.report_scores()
    # metrics.report_confusion_matrix()
