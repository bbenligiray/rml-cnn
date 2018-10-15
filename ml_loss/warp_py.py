"""
Gong, Yunchao, et al. "Deep convolutional ranking for multilabel image
annotation." arXiv preprint arXiv:1312.4894 (2013). Section 2.2.3
"""
import os

import tensorflow as tf
from tensorflow.python.framework import ops # pylint: disable=E0611


NORM_FACTOR = 20

real_path = os.path.dirname(os.path.realpath(__file__))
warp_module = tf.load_op_library(os.path.join(real_path, 'warp.so'))
warp_grad_module = tf.load_op_library(os.path.join(real_path, 'warp_grad.so'))
@ops.RegisterGradient("Warp")
def warp_grad(op, grad):
    grad_mult = warp_grad_module.warp_grad(op.inputs[0], op.inputs[1]) / NORM_FACTOR
    return [None, grad * grad_mult]
def warp(y_true, y_pred):
    return warp_module.warp(y_true, y_pred) / NORM_FACTOR
