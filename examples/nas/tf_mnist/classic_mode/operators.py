import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D #(AveragePooling2D, BatchNormalization, Conv2D, Dense, MaxPool2D)
import math


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def sum_op(inputs):
    """sum_op"""
    fixed_input = inputs[0][0]
    optional_input = inputs[1][0]
    fixed_shape = fixed_input.get_shape().as_list()
    optional_shape = optional_input.get_shape().as_list()
    assert fixed_shape[1] == fixed_shape[2]
    assert optional_shape[1] == optional_shape[2]
    pool_size = math.ceil(optional_shape[1] / fixed_shape[1])
    pool_out = tf.nn.avg_pool(optional_input, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding='SAME')
    conv_matrix = weight_variable([1, 1, optional_shape[3], fixed_shape[3]])
    conv_out = tf.nn.conv2d(pool_out, conv_matrix, strides=[1, 1, 1, 1], padding='SAME')
    return fixed_input + conv_out


def conv2d(out_ch=-1, size=-1):
    """conv2d returns a 2d convolution layer with full stride."""
    #if not inputs[1]:
    #    x_input = inputs[0][0]
    #else:
    #    x_input = sum_op(inputs)
    if size in [1, 3]:
        return Conv2D(out_ch, size, padding='same')
    else:
        raise Exception("Unknown filter size: %d." % size)

def twice_conv2d(out_ch=-1, size=-1):
    """twice_conv2d"""
    #if not inputs[1]:
    #    x_input = inputs[0][0]
    #else:
    #    x_input = sum_op(inputs)
    if size in  [3, 7]:
        conv1 = Conv2D(int(out_ch/2), size, padding='same')
        conv2 = Conv2D(out_ch, size, padding='same')
        return tf.keras.Sequential([conv1, conv2])
    else:
        raise Exception("Unknown filter size: %d." % size)

def separable_conv(out_ch=-1, size=-1):
    """separable_conv"""
    #if not inputs[1]:
    #    x_input = inputs[0][0]
    #else:
    #    x_input = sum_op(inputs)
    if size in [3, 5, 7]:
        depth_matrix = weight_variable([size, size, in_ch, 1])
        point_matrix = weight_variable([1, 1, 1*in_ch, out_ch])
        return SeparableConv2D(out_ch, size, strides=(1, 1), padding='SAME')
    else:
        raise Exception("Unknown filter size: %d." % size)


def avg_pool(inputs, size=-1):
    """avg_pool downsamples a feature map."""
    if not inputs[1]:
        x_input = inputs[0][0]
    else:
        x_input = sum_op(inputs)
    if size in [3, 5, 7]:
        return tf.nn.avg_pool(x_input, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')
    else:
        raise Exception("Unknown filter size: %d." % size)

def max_pool(inputs, size=-1):
    """max_pool downsamples a feature map."""
    #if not inputs[1]:
    #    x_input = inputs[0][0]
    #else:
    #    x_input = sum_op(inputs)
    if size in [3, 5, 7]:
        return tf.nn.max_pool(x_input, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')
    else:
        raise Exception("Unknown filter size: %d." % size)


def post_process(inputs, ch_size=-1):
    """post_process"""
    x_input = inputs[0][0]
    bias_matrix = bias_variable([ch_size])
    return tf.nn.relu(x_input + bias_matrix)
