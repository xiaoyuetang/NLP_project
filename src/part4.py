import os

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from part1 import Feature
from part3 import Train

def numpy_to_dict(w, f, reverse = False):
    '''
    Converts a numpy array w to a dictionary with keys from f.
    '''
    for i,k in enumerate(f.keys()):
        f[k] = w[i]
    return f


def prepare_grad_for_bfgs(grads,f):
    '''
    Converts a dictionary to a numpy array.
    '''
    np_grads = np.zeros(len(f))
    for i,k in enumerate(f.keys()):
        np_grads[i] = grads[k]
    return np_grads


def callbackF(w):
    '''
    This function will be called by "fmin_l_bfgs_b"
    Arg:
        w: weights, numpy array
    '''
    loss = compute_crf_loss(train_inputs,train_labels,f,states,0.1,regularization=True)
    print('Loss:{0:.4f}'.format(loss))

def get_loss_grad(w,*args):
    '''
    This function will be called by "fmin_l_bfgs_b"
    Arg:
        w: weights, numpy array
    Returns:
        loss: loss, float
        grads: gradients, numpy array
    '''

    train_inputs,train_labels,f,states = args
    f = numpy_to_dict(w,f)
    # compute loss and grad
    loss = train.loss_function(train_inputs, train_labels, f, 0.1, reg=True)
    grads = train.calculate_gradients(train_inputs, train_labels, f, 0.1, reg=True)
    grads = prepare_grad_for_bfgs(grads, f)
    # return loss and grad
    return loss, grads


if __name__ == '__main__':
    dataset = os.path.join(os.path.dirname(__file__), "..", "data", "partial")
    train_path = os.path.join(dataset, "train")
    train = Train(train_path)
    sentence_list, tag_list = train.get_sentences_tags(train_path)
    feature = Feature(train_path)
    init_w = np.zeros(len(feature.feature_dict))
    result = fmin_l_bfgs_b(get_loss_grad, init_w, args=(sentence_list, tag_list, feature.feature_dict, feature.tags), pgtol=0.01, callback=callbackF)



# def callbackF(w):
#     '''
#     This function will only be called by "fmin_l_bfgs_b"
#     Arg:
#     w: weights, numpy array
#     '''
#
#     loss = get_loss_grad(w)[0]
#     print('Loss:{0:.4f}'.format(loss))
#
#
# def get_loss_grad(w, *args):
#     '''
#     This function will only be called by "fmin_l_bfgs_b"
#     Arg:
#     w: weights, numpy array
#     Returns:
#     6
#     loss: loss, float
#     grads: gradients, numpy array
#     '''
#     # convert np array to dict
#     sentence_list, tag_list, feature_dic = args
#     for i, k in enumerate(feature_dic.keys()):
#         feature_dic[k] = w[i]
#
#     return loss, grads
#
# result = fmin_l_bfgs_b(get_loss_grad, init_w, pgtol=0.01, callback=callbackF)


