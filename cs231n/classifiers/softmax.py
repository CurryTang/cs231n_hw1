import numpy as np
from random import shuffle

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

  dimension = W.shape[0]
  num_classes = W.shape[1]
  num_examples = X.shape[0]


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_examples):
    example = X[i]
    scores = example.dot(W)
    ## compute the softmax loss
    ground_truth_label = y[i]
    scores -= np.max(scores)
    p = np.exp(scores) / np.sum(np.exp(scores))
    for k in range(num_classes):
      if (ground_truth_label == k):
        dW[:, y[i]] += (p[k] - 1) * X[i, :]
      else:
        dW[:, k] += (p[k]) * X[i, :]
    L_i = -np.log(p[ground_truth_label])
    loss += L_i

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_examples
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_examples
  dW += reg * W

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
  dimension = W.shape[0]
  num_classes = W.shape[1]
  num_examples = X.shape[0]
  scores = X.dot(W)
  p_mat = np.exp(scores) / np.sum(np.exp(scores), axis = 1, keepdims = True)
  correct_log_probs = -np.log(p_mat[range(num_examples), y])
  data_loss = np.sum(correct_log_probs) / num_examples
  loss = data_loss + 0.5 * reg * np.sum(W * W)

  dscore = p_mat
  dscore[range(num_examples), y] -= 1
  dscore /= num_examples
  dW = np.dot(X.T, dscore)
  dW += reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

