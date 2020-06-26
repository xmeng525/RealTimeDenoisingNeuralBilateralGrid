"""
description: network architecture

@author: Xiaoxu Meng
"""

import tensorflow as tf
import numpy as np
import math

def lrelu(x, leak=0.2, name="my_relu"):
    with tf.variable_scope(name):
        return tf.nn.leaky_relu(x, leak, name)

def conv_layer(batch_input, filter_size, stride, n_output_ch):
    input_shape = batch_input.get_shape().as_list()
    print('DEBUG:: InShape = ', input_shape)
    n_input_ch = input_shape[3]
    
    with tf.name_scope('conv'):
        W = tf.get_variable(name='weight',
            shape=[filter_size, filter_size, n_input_ch, n_output_ch], dtype=np.float32)
        tf.summary.histogram('weight', W)
        b = tf.get_variable(name='bias',shape=[n_output_ch], dtype=np.float32)
        tf.summary.histogram('bias', b)
        output = tf.add(tf.nn.conv2d(
            batch_input, W, strides=stride, padding='SAME'), b, name='Wx_p_b')
        tf.summary.histogram('Wx_p_b', output)
    print('DEBUG:: OutShape = ', output.get_shape().as_list())
    return output

def conv_layer_dilated(batch_input, filter_size, rate, n_output_ch):
    input_shape = batch_input.get_shape().as_list()
    print('DEBUG:: InShape = ', input_shape)
    n_input_ch = input_shape[3]

    with tf.name_scope('conv_dilated'):
        output = tf.add(tf.nn.atrous_conv2d(
            batch_input, W, rate=rate, padding='SAME'), b, name='Wx_p_b')
        tf.summary.histogram('weight', W)
        tf.summary.histogram('bias', b)
        tf.summary.histogram('Wx_p_b', output)
    print('DEBUG:: OutShape = ', output.get_shape().as_list())
    return output, W
