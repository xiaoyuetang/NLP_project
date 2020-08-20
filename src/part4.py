from scipy.optimize import fmin_l_bfgs_b

def callbackF(w):
    '''
    This function will only be called by "fmin_l_bfgs_b"
    Arg:
    w: weights, numpy array
    '''
    loss = get_loss_grad(w)[0]
    print('Loss:{0:.4f}'.format(loss))

def get_loss_grad(w):
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
    return loss, grads

result = fmin_l_bfgs_b(get_loss_grad, init_w, pgtol=0.01, callback=callbackF)