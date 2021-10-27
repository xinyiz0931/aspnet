"""
`train_helper.py`
Generate categorical inputs/labels for training. 
Author: xinyi
Date: 2021/7/26
Derived from: tensorflow 2.5.0
"""

import numpy as np
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.utils.to_categorical')
def positive_to_range_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    >>> a = positive_to_range_categorical([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    tf.Tensor(
      [[1. 1. 1. 1.]
       [0. 1. 1. 1.]
       [0. 0. 1. 1.]
       [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    for i in range(n):
        categorical[i, int(y[i]):, ] = 1
    # categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


@keras_export('keras.utils.to_categorical')
def negative_to_range_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    >>> a = negative_to_range_categorical([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    tf.Tensor(
      [[1. 0. 0. 0.]
       [1. 1. 0. 0.]
       [1. 1. 1. 0.]
       [1. 1. 1. 1.]], shape=(4, 4), dtype=float32)
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    for i in range(n):
        categorical[i, :int(y[i]+1)] = 1
    # categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

if __name__ == '__main__':
    a = negative_to_range_categorical(0, num_classes=4)
    a = negative_to_range_categorical(3, num_classes=4)
    print(a)