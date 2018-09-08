import os

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops


real_path = os.path.dirname(os.path.realpath(__file__))
robust_warp_sup_module = tf.load_op_library(os.path.join(real_path, 'robust_warp_sup.so'))
robust_warp_sup_grad_module = tf.load_op_library(os.path.join(real_path, 'robust_warp_sup_grad.so'))
@ops.RegisterGradient("RobustWarpSup")
def robust_warp_sup_grad(op, grad):
  grad_mult = robust_warp_sup_grad_module.robust_warp_sup_grad(op.inputs[0], op.inputs[1])
  return [None, grad * grad_mult]
def robust_warp_sup(y_true, y_pred):
  return robust_warp_sup_module.robust_warp_sup(y_true, y_pred)