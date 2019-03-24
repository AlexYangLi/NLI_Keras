# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: weight_average.py

@time: 2019/3/14 10:43

@desc:

"""

import tensorflow as tf
from keras import backend as K, regularizers, initializers
from keras.engine.topology import Layer


class WeightedAverage(Layer):
    def __init__(self, l2_coef=None, do_layer_norm=False, scale=True, **kwargs):
        self.l2_coef = l2_coef
        self.do_layer_norm = do_layer_norm
        self.scale = scale
        self.num_layers = None
        super(WeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_layers = input_shape[1]
        self.W = self.add_weight(shape=(self.num_layers, ), initializer=initializers.Zeros(),
                                 regularizer=regularizers.get(regularizers.l2(self.l2_coef)),
                                 name='{}_w'.format(self.name))
        if self.scale:
            self.gamma = self.add_weight(shape=(1, ), initializer=initializers.Ones(), name='{}_gamma'.format(self.name))
        super(WeightedAverage, self).build(input_shape)

    def call(self, inputs):
        # refer to: https://github.com/allenai/bilm-tf/blob/master/bilm/elmo.py
        # normalize the weights
        norm_weights = tf.split(K.softmax(self.W + 1.0 / self.num_layers), self.num_layers) # keras doesn't have K.split
        # split layers
        layers = tf.split(inputs, self.num_layers, axis=1)

        pieces = []
        for w, l in zip(norm_weights, layers):
            if self.do_layer_norm:
                pieces.append(w * self.layer_norm(K.squeeze(l, axis=1)))
            else:
                pieces.append(w * K.squeeze(l, axis=1))
        sum_pieces = tf.add_n(pieces)
        if self.scale:
            sum_pieces *= self.gamma
        return sum_pieces

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0:1] + input_shape[2:])

    @staticmethod
    def layer_norm(l):
        mask = K.cast(K.not_equal(l, 0), dtype=tf.float32)
        N = K.sum(mask)
        mean = K.sum(l) / N
        variance = K.sum(K.square((l - mean) * mask))
        return K.batch_normalization(l, mean, variance, None, None, epsilon=1E-12)
