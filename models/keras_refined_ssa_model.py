# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_refined_ssa_model.py

@time: 2019/4/8 22:46

@desc: refined structured self attention model

"""

import keras.backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import Bidirectional, LSTM, concatenate, subtract, multiply, GlobalMaxPooling1D, \
    GlobalAveragePooling1D, Dense
from models.keras_base_model import KerasBaseModel
from layers.attention import MultiSelfAttention, DotProductAttention


class KerasRefinedSSAModel(KerasBaseModel):
    """refined structured self attention model"""
    def __init__(self, config, **kwargs):
        super(KerasRefinedSSAModel, self).__init__(config, **kwargs)

    def build(self, input_config='token', elmo_output_mode='elmo', elmo_trainable=None, elmo_model_url=None,
              add_penalty=False, penalty_coef=0.3):
        inputs, premise_embed, hypothesis_embed = self.build_input(input_config=input_config, mask_zero=True,
                                                                   elmo_output_mode=elmo_output_mode,
                                                                   elmo_trainable=elmo_trainable,
                                                                   elmo_model_url=None)

        bilstm = Bidirectional(LSTM(units=self.config.rnn_units, return_sequences=True))
        premise_hidden = bilstm(premise_embed)
        hypothesis_hidden = bilstm(hypothesis_embed)

        self_attention = MultiSelfAttention(return_weight=add_penalty)
        # use multi hops of attention to extract sentence embedding
        if add_penalty:
            p_attend_weight, p_self_attend = self_attention(premise_hidden)
            h_attend_weight, h_self_attend = self_attention(hypothesis_hidden)
        else:
            p_self_attend = self_attention(premise_hidden)
            h_self_attend = self_attention(hypothesis_hidden)

        # extract interactive attention between tow sentence
        p_inter_attend, h_inter_attend = DotProductAttention()([p_self_attend, h_self_attend])

        p_enhance = concatenate([p_self_attend, p_inter_attend, subtract([p_self_attend, p_inter_attend]),
                                 multiply([p_self_attend, p_inter_attend])])
        h_enhance = concatenate([h_self_attend, h_inter_attend, subtract([h_self_attend, h_inter_attend]),
                                 multiply([h_self_attend, h_inter_attend])])

        p_avg = GlobalAveragePooling1D()(p_enhance)
        p_max = GlobalMaxPooling1D()(p_enhance)
        h_avg = GlobalAveragePooling1D()(h_enhance)
        h_max = GlobalMaxPooling1D()(h_enhance)

        if self.config.add_features:
            p_concat_h = concatenate([p_avg, p_max, h_avg, h_max, inputs[-1]])
        else:
            p_concat_h = concatenate([p_avg, p_max, h_avg, h_max])

        dense = Dense(units=300, activation='tanh')(p_concat_h)
        output = Dense(self.config.n_class, activation='softmax')(dense)

        model = Model(inputs, output)
        if add_penalty:
            loss = self.penalty_loss(p_attend_weight, h_attend_weight, penalty_coef)
        else:
            loss = 'categorical_crossentropy'
        model.compile(loss=loss, metrics=['acc'], optimizer=self.config.optimizer)
        return model

    @staticmethod
    # About How to add a penalty term to loss function in Keras?
    # See: https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
    # Also see: https://github.com/keras-team/keras/issues/2121
    def penalty_loss(p_attend_weight, h_attend_weight, penalty_coef):
        def penalty_term(a):
            a_t = K.permute_dimensions(a, (0, 2, 1))
            a_mul_at = K.batch_dot(a, a_t, axes=(2, 1))
            batch_size = K.int_shape(a)[0]
            n_hop = K.int_shape(a)[1]
            batch_identity = K.reshape(K.tile(K.eye(n_hop), [batch_size, 1]), [-1, n_hop, n_hop])
            return K.square(tf.norm(a_mul_at - batch_identity, ord='fro', axis=[-2, -1]))

        def loss(y_true, y_pred):
            return K.categorical_crossentropy(y_true, y_pred) + \
                   (penalty_term(p_attend_weight) + penalty_term(h_attend_weight)) * penalty_coef

        return loss
