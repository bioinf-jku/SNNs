# -*- coding: utf-8 -*-
'''
Tensorflow Implementation of the Scaled ELU function and Dropout
'''

from __future__ import absolute_import, division, print_function
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
import numpy as np
import tensorflow as tf

# (1) scale inputs to zero mean and unit variance


# (2) use SELUs
def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


# (3) initialize weights with stddev sqrt(1/n)
# e.g. use:
initializer = layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')


# (4) use this dropout
def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = math_ops.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * math_ops.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))


def get_timestamp(fmt='%y%m%d_%H%M'):
    '''Returns a string that contains the current date and time.

    Suggested formats:
        short_format=%y%m%d_%H%M  (default)
        long format=%Y%m%d_%H%M%S
    '''
    import datetime
    now = datetime.datetime.now()
    return datetime.datetime.strftime(now, fmt)


def generate_slices(n, slice_size, allow_smaller_final_batch=True):
    """Generates slices of given slice_size up to n"""
    start, end = 0, 0
    for pack_num in range(int(n / slice_size)):
        end = start + slice_size
        yield slice(start, end, None)
        start = end
    # last slice might not be a full batch
    if allow_smaller_final_batch:
        if end < n:
            yield slice(end, n, None)


def generate_minibatches(batch_size, ph_list, data_list, n_epochs=1,
                         allow_smaller_final_batch=False, shuffle=True,
                         feed_dict=None):
    cnt_epochs = 0
    assert len(ph_list) == len(data_list), "Passed different number of data and placeholders"
    assert len(data_list) >= 0, "Passed empty lists"

    n_samples = data_list[0].shape[0]
    n_items = len(data_list)

    while True:
        if shuffle:
            idx = np.arange(n_samples)
            np.random.shuffle(idx)
            for i in range(n_items):
                data_list[i] = data_list[i][idx]

        if feed_dict is None:
                feed_dict = {}

        for s in generate_slices(n_samples, batch_size, allow_smaller_final_batch):
            for i in range(n_items):
                ph = ph_list[i]
                d = data_list[i][s]
                feed_dict[ph] = d
            yield feed_dict
        cnt_epochs += 1
        if n_epochs is not None and cnt_epochs >= n_epochs:
            break
