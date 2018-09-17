"""
Gong, Yunchao, et al. "Deep convolutional ranking for multilabel image
annotation." arXiv preprint arXiv:1312.4894 (2013). Section 2.2.2
"""


import tensorflow as tf

norm_factor = 20


def pairwise_ranking(y_true, y_pred):
  tf_batch_size = tf.shape(y_pred)[0]
  tf_no_classes = tf.shape(y_pred)[1]

  y_false = 1 - y_true

  # condition for broadcasting
  y_true_pos = tf.reshape(y_true, [tf_batch_size, tf_no_classes, 1])
  y_true_neg = tf.reshape(y_false, [tf_batch_size, 1, -1])

  # find indices where one is true and the other is false
  clash_indices = tf.multiply(y_true_pos, y_true_neg)
  clash_indices = tf.clip_by_value(clash_indices, 0, 1)

  # calculate scores that is going to contribute to the loss
  y_true_scores = y_true * y_pred
  y_false_scores = y_false * y_pred

  # condition the scores for broadcasting
  y_true_scores_bc = tf.reshape(y_true_scores, [tf_batch_size, tf_no_classes, 1])
  y_false_sores_bc = tf.reshape(y_false_scores, [tf_batch_size, 1, -1])

  loss_matrix = -y_true_scores_bc + y_false_sores_bc
  loss_matrix = tf.maximum(tf.zeros_like(loss_matrix), 1 + loss_matrix)

  effective_loss_matrix = loss_matrix * clash_indices

  loss = tf.reduce_sum(effective_loss_matrix) / tf.cast(tf_batch_size, tf.float32)

  return loss / norm_factor