"""
Gong, Yunchao, et al. "Deep convolutional ranking for multilabel image
annotation." arXiv preprint arXiv:1312.4894 (2013). Section 2.2.1
"""


import tensorflow as tf


def ml_softmax(y_true, y_pred):
  y_pred = tf.nn.softmax(y_pred)
  # requires a large epsilon
  y_pred_log = tf.log(y_pred + 1e-3)
  dot_product = tf.reduce_sum(tf.multiply(y_pred_log, y_true), axis=1)
  norm = tf.reduce_sum(y_true, axis=1)
  norm_dot_product = tf.div(dot_product, norm)
  return -tf.reduce_mean(norm_dot_product)