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
        a = np.exp(f)
        a_sum = a.sum()
        r = a[y[k]]/a.sum()
        loss -= np.log(r)

        # forward propagation
        for j in xrange(W.shape[1]): # classes loop
            c = (int(j==y[k])*a_sum-a[y[k]])/(a[y[k]]*a_sum)
            for i in xrange(W.shape[0]):  # features loop
                dW[i,j] += -X[k,i]*a[j]*c

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

    Inputs and outputs are the same as softmax_loss_naive.
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

    cls = np.arange(10)

    for k in xrange(num_train):
        # forward propagation
        f = W.T.dot(X[k, :])
        a = np.exp(f)
        a = a[:,np.newaxis]
        a_sum = a.sum()
        r = a[y[k]] / a.sum()
        loss -= np.log(r)

        sam = X[k, :]
        # backward propagation
        c = ((cls == y[k]).astype(int)*a_sum - a[y[k]])/(a[y[k]] * a_sum)
        c = c[:,np.newaxis]
        c = np.multiply(c, a)
        dW += - np.dot(sam[:,np.newaxis],c.T)

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

