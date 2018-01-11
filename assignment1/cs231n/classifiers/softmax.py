import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]

    print('Test')
    for k in xrange(num_train):
        # forward propagation
        f = W.T.dot(X[k,:])
        e = np.exp(f)
        e_sum = e.sum()
        r = e[y[k]]/e.sum()
        loss -= np.log(r)

        # backward propagation
        for j in xrange(W.shape[1]): # classes loop
            #c = (int(j==y[k])*a_sum-a[y[k]])/(a[y[k]]*a_sum)
            df = e[j]/e_sum - int(j==y[k])

            for i in xrange(W.shape[0]):  # features loop
                # dw = X[k,i]
                dW[i,j] += df*X[k,i]

    loss /=num_train
    loss +=reg*np.sum(W*W)
    dW /=num_train
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]

    # forward propagation
    f = X.dot(W)
    e = np.exp(f)
    e_sum = np.sum(e, axis=1, keepdims=True)
    idx = np.ravel_multi_index([range(0, num_train), y], [num_train, num_classes])
    correct_class_ex = e.take(idx)
    correct_class_ex = correct_class_ex[:,np.newaxis]
    L = -np.log(np.divide(correct_class_ex, e_sum))
    loss = np.mean(L)+0.5*reg*np.sum(W*W)

    # backward propagation
    # dL/df
    cls = np.tile(np.arange(10), [num_train, 1])
    delta = (cls == y[:, np.newaxis]).astype(int)
    df = np.divide(e,e_sum) - delta
    # df/dw
    dw = X.T
    # dL/dw
    dW = dw.dot(df)
    dW += reg*W
    dW /= num_train
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

