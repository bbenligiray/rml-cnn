import os

import tensorflow as tf
from tensorflow.python.framework import ops # pylint: disable=E0611


NORM_FACTOR = 20

real_path = os.path.dirname(os.path.realpath(__file__))
robust_warp_module = tf.load_op_library(os.path.join(real_path, 'robust_warp.so'))
robust_warp_grad_module = tf.load_op_library(os.path.join(real_path, 'robust_warp_grad.so'))
@ops.RegisterGradient("RobustWarp")
def robust_warp_grad(op, grad):
    grad_mult = robust_warp_grad_module.robust_warp_grad(op.inputs[0], op.inputs[1]) / NORM_FACTOR
    return [None, grad * grad_mult]
def robust_warp(y_true, y_pred):
    return robust_warp_module.robust_warp(y_true, y_pred) / NORM_FACTOR
