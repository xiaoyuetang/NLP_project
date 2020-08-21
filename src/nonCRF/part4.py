import os

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from part1 import Feature
from part3 import Train


class Learn():
    def __init__(self):
        self.feature = Feature(train_path)
        self.feature_dict = self.feature.feature_dict
        self.sentence_list, self.tag_list = train.get_sentences_tags(train_path)
        self.lr = 0.1
    def callbackF(self, w):
        '''
        This function will only be called by "fmin_l_bfgs_b"
        Arg:
        w: weights, numpy array
        '''
        print("in callbackf")
        loss = self.get_loss_grad(w)[0]
        print('Loss:{0:.4f}'.format(loss))

    def get_loss_grad(self, w):
        '''
        This function will only be called by "fmin_l_bfgs_b"
        Arg:
            w: weights, numpy array
        Returns:
            loss: loss, float
            grads: gradients, numpy array
        '''
        # to be completed by you,
        # based on the modified loss and gradients,
        # with L2 regularization included
        print("calculate gradient")
        # gradient_dic = train.calculate_gradients(self.sentence_list, self.tag_list)
        # grads = np.zeros_like(w)
        # for i, k in enumerate(self.feature_dict.keys()):
        #     grads[i] = gradient_dic[k]

        for i, k in enumerate(self.feature_dict.keys()):
            self.feature_dict[k] = w[i]
        grad_dic = train.calculate_gradients(self.sentence_list, self.tag_list, self.feature_dict, lr=0.1, reg=True)
        grads = np.zeros_like(w)
        for i, k in enumerate(self.feature_dict.keys()):
            grads[i] = grad_dic[k]
        print("***gradients: ", grads)

        loss = train.loss_function(self.sentence_list, self.tag_list, self.feature_dict, lr=0.1, reg=True)
        print("***loss: ", loss)

        return loss, grads


if __name__ == '__main__':
    dataset = os.path.join(os.path.dirname(__file__), "..", "data", "partial")
    train_path = os.path.join(dataset, "train")
    train = Train(train_path)
    learn = Learn()
    init_w = np.zeros(len(learn.feature_dict))
    # result = fmin_l_bfgs_b(learn.get_loss_grad, init_w, args=(sentence_list, tag_list, lr), pgtol=0.01, callback=learn.callbackF(init_w, sentence_list, tag_list, lr))
    # result = fmin_l_bfgs_b(learn.get_loss_grad(init_w, sentence_list, tag_list, lr=0.1), init_w, pgtol=0.01, callback=learn.callbackF(init_w, sentence_list, tag_list, lr=0.1))
    result = fmin_l_bfgs_b(learn.get_loss_grad, init_w, pgtol=0.01, callback=learn.callbackF)
    print(result)