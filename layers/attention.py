# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: attention.py

@time: 2019/2/3 20:43

@desc: all kinds of attention mechanism, supporting masking

"""

import numpy as np
from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer

import tensorflow as tf


class SelfAttention(Layer):
    """
    self-attention mechanism, supporting masking
    """
    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None, W_constraint=None,
                 u_constraint=None, b_constraint=None, **kwargs):
        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),  initializer=self.init,
                                     name='{}_W'.format(self.name), regularizer=self.W_regularizer,
                                     constraint=self.W_constraint)

        self.b = self.add_weight(shape=(input_shape[-1],), initializer='zero', name='{}_b'.format(self.name),
                                 regularizer=self.b_regularizer, constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],), initializer=self.init, name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer, constraint=self.u_constraint)

        super(SelfAttention, self).build(input_shape)

    def compute_mask(self, inputs, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, inputs, mask=None):
        x = K.tanh(K.dot(inputs, self.W) + self.b)

        ait = SelfAttention.dot_product(x, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        attend_output = K.sum(inputs * K.expand_dims(a), axis=1)
        return attend_output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    @staticmethod
    def dot_product(x, kernel):
        """
        Wrapper for dot product operation, in order to be compatible with both
        Theano and Tensorflow
        Args:
            x (): input
            kernel (): weights
        Returns:
        """
        if K.backend() == 'tensorflow':
            return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
        else:
            return K.dot(x, kernel)


class DotProductAttention(Layer):
    """
    dot-product-attention mechanism, supporting masking
    """
    def __init__(self, return_attend_weight=False, keep_mask=True, **kwargs):
        self.return_attend_weight = return_attend_weight
        self.keep_mask = keep_mask
        self.supports_masking = True
        super(DotProductAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        input_shape_a, input_shape_b = input_shape

        if len(input_shape_a) != 3 or len(input_shape_b) != 3:
            raise ValueError('Inputs into DotProductAttention should be 3D tensors')

        if input_shape_a[-1] != input_shape_b[-1]:
            raise ValueError('Inputs into DotProductAttention should have the same dimensionality at the last axis')

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        inputs_a, inputs_b = inputs

        if mask is not None:
            mask_a, mask_b = mask
        else:
            mask_a, mask_b = None, None

        e = K.exp(K.batch_dot(inputs_a, inputs_b, axes=2))  # similarity between a & b

        # apply mask before normalization (softmax)
        if mask_a is not None:
            e *= K.expand_dims(K.cast(mask_a, K.floatx()), 2)
        if mask_b is not None:
            e *= K.expand_dims(K.cast(mask_b, K.floatx()), 1)

        e_b = e / K.cast(K.sum(e, axis=2, keepdims=True) + K.epsilon(), K.floatx())    # attention weight over b
        e_a = e / K.cast(K.sum(e, axis=1, keepdims=True) + K.epsilon(), K.floatx())     # attention weight over a

        if self.return_attend_weight:
            return [e_b, e_a]

        a_attend = K.batch_dot(e_b, inputs_b, axes=(2, 1))  # a attend to b
        b_attend = K.batch_dot(e_a, inputs_a, axes=(1, 1))  # b attend to a
        return [a_attend, b_attend]

    def compute_mask(self, inputs, mask=None):
        if self.keep_mask:
            return mask
        else:
            return [None, None]

    def compute_output_shape(self, input_shape):
        if self.return_attend_weight:
            input_shape_a, input_shape_b = input_shape
            return [(input_shape_a[0], input_shape_a[1], input_shape_b[1]),
                    (input_shape_a[0], input_shape_a[1], input_shape_b[1])]
        return input_shape


class IntraSentenceAttention(Layer):
    """
    intra-sentence-attention mechanism, supporting masking
    """

    def __init__(self, return_attend_weight=False, **kwargs):
        self.return_attend_weight = return_attend_weight
        self.distance_term = None
        self.supports_masking = True
        super(IntraSentenceAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Input into IntraSentenceAttention should be a 3D tensor')
        # create distance-sensitive bias term
        time_steps = input_shape[1]
        distance_term = np.zeros(shape=(time_steps, time_steps))
        for i in range(time_steps):
            for j in range(time_steps):
                distance_term[i][j] = min(i-j, 10)
        self.distance_term = K.variable(distance_term)

    def call(self, inputs, mask=None):

        e = K.exp(K.batch_dot(inputs, inputs, axes=2) + self.distance_term)

        # apply mask before normalization (softmax)
        if mask is not None:
            e *= K.expand_dims(K.cast(mask, K.floatx()), 2)
            e *= K.expand_dims(K.cast(mask, K.floatx()), 1)

        # normalization
        e = e / K.cast(K.sum(e, axis=-1, keepdims=True) + K.epsilon(), K.floatx())  # attention weight over b

        if self.return_attend_weight:
            return e

        attend = K.batch_dot(e, inputs, axes=(2, 1))
        return attend

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        if self.return_attend_weight:
            return input_shape[0], input_shape[1], input_shape[1]
        return input_shape


class MultiSelfAttention(Layer):
    """
    multiple self-attention mechanism, supporting masking
    see "Lin et al. A Structured Self-Attentive Sentence Embedding" for more details (section 2)
    """
    def __init__(self, n_hop=30, return_weight=True, W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None, **kwargs):
        self.supports_masking = True

        self.n_hop = n_hop
        self.return_weight = return_weight

        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        super(MultiSelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),  initializer=self.init,
                                     name='{}_W'.format(self.name), regularizer=self.W_regularizer,
                                     constraint=self.W_constraint)

        self.b = self.add_weight(shape=(input_shape[-1],), initializer='zero', name='{}_b'.format(self.name),
                                 regularizer=self.b_regularizer, constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1], self.n_hop), initializer=self.init, name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer, constraint=self.u_constraint)

        super(MultiSelfAttention, self).build(input_shape)

    def compute_mask(self, inputs, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        x = K.tanh(K.dot(x, self.W) + self.b)   # [batch_size, time_step, embed_dim]
        ait = K.dot(x, self.u)  # perform multiple hops of attentions   # [batch_size, time_step, n_hop]
        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.expand_dims(K.cast(mask, K.floatx()), axis=-1)

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())  # [batch_size, time_step, n_hop]

        attend_output = K.batch_dot(a, x, axes=1)   # [batch_size, n_hop, embed_dim]
        if self.return_weight:
            a = K.permute_dimensions(x, (0, 2, 1))  # [batch_size, n_hop, time_step]
            return [a, attend_output]
        else:
            return attend_output

    def compute_output_shape(self, input_shape):
        if self.return_weight:
            return [(input_shape[0], self.n_hop, input_shape[1]),
                    (input_shape[0], self.n_hop, input_shape[-1])]
        else:
            return input_shape[0], self.n_hop, input_shape[-1]











