import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).
  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  :param
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  :return
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  # gradient
  L = np.zeros([num_classes, 1])

  # loop over training samples
  for i in xrange(num_train):
    scores = X[i].dot(W) # score of the given sample
    correct_class_score = scores[y[i]] # get weight of the correct score
    margin = np.maximum(0, scores - correct_class_score+1)
    loss = margin.mean()

    # calculate gradient
    for j in xrange(num_classes): # loop over classes
        if j == y[i]: # correct class
            dW[:,j] -=X[i]*(margin>0).astype(int).sum()
        else: #incorrect class
            dW[:,j] +=X[i]*int(margin[j]>0)

  dW = dW/(num_classes*num_train)

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  dW = np.zeros(W.shape)  # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  # gradient
  L = np.zeros([num_classes, 1])

  # loop over training samples
  #for i in xrange(num_train):
  #    scores = W.T.dot(X[i]) #.dot(W)  # score of the given sample
  #    correct_class_score = scores[y[i]]  # get weight of the correct score
  #    margin = np.maximum(0, scores - correct_class_score + 1)
  #    loss = margin.mean()

      # calculate gradient
  #    c = (margin > 0).astype(int)
  #    c[y[i]] = -(margin > 0).astype(int).sum()
  #    tmp = X[i,:]
  #    dW += tmp[:,np.newaxis].dot(c[:,np.newaxis].T)

  scores = X.dot(W)  # .dot(W)  # score of the given sample
  idx = np.ravel_multi_index([range(0, num_train ), y], [num_train, num_classes])
  correct_class_score = scores.take(idx)  # get weight of the correct score
  margin = np.maximum(0, scores - correct_class_score[:,np.newaxis] + 1)

  loss = margin.mean()

  # calculate gradient
  c = (margin > 0).astype(int) # indicator function
  c[:,y] = -(margin > 0).astype(int).sum(axis=1)

  #print(c)

  tmp = X[i, :]
  dW += tmp[:, np.newaxis].dot(c[:, np.newaxis].T)


  dW = dW / (num_classes * num_train)

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  return loss, dW
