#!/usr/bin/env python
"""
Loss functions.

"""


import tensorflow as tf
import semver


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    
    diff = tf.abs(y_true - y_pred)
    condition = tf.less_equal(diff, max_grad)
    diff1 = 0.5 * tf.square(diff)
    diff2 = max_grad * diff - 0.5 * tf.square(max_grad)
    loss = tf.where(condition, diff1, diff2)
    return loss
    


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    return tf.reduce_mean(huber_loss(y_true, y_pred, max_grad))
    
    
