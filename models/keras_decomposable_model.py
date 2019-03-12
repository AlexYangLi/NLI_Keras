# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_decomposable_model.py

@time: 2019/2/6 22:37

@desc:

"""

from keras.layers import  LSTM, Bidirectional, GlobalAveragePooling1D, \
      concatenate, Lambda, subtract, multiply, Dense, TimeDistributed, Dropout
import keras.backend as K
from keras import Model

from models.keras_base_model import KerasBaseModel
from layers.attention import DotProductAttention, IntraSentenceAttention


class KerasDecomposableAttentionModel(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(KerasDecomposableAttentionModel, self).__init__(config, **kwargs)

    def build(self, input_config='token', elmo_output_mode='elmo', elmo_trainable=None, add_intra_sentence_attention=False):
        inputs, premise_embed, hypothesis_embed = self.build_input(input_config=input_config, mask_zero=True,
                                                                   elmo_output_mode=elmo_output_mode,
                                                                   elmo_trainable=elmo_trainable)

        # input representation
        if add_intra_sentence_attention:
            premise_intra = IntraSentenceAttention()(premise_embed)
            hypothesis_intra = IntraSentenceAttention()(hypothesis_embed)
            premise_input = concatenate([premise_embed, premise_intra])
            hypothesis_input = concatenate([hypothesis_embed, hypothesis_intra])
        else:
            premise_input = premise_embed
            hypothesis_input = hypothesis_embed

        # attend
        f1 = TimeDistributed(Dense(units=200, activation='relu'))
        f2 = TimeDistributed(Dense(units=200, activation='relu'))
        premise_f = f2(f1(premise_input))
        hypothesis_f = f2(f1(premise_input))

        hypothesis_attend_w, premise_attend_w = DotProductAttention(return_attend_weight=True)([premise_f, hypothesis_f])
        hypothesis_attend = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(2, 1)))([hypothesis_attend_w, hypothesis_input])
        premise_attend = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(1, 1)))([premise_attend_w, premise_input])

        # compare
        g1 = TimeDistributed(Dense(units=200, activation='relu'))
        g2 = TimeDistributed(Dense(units=200, activation='relu'))
        premise_g = g2(g1(concatenate([premise_input, hypothesis_attend])))
        hypothesis_g = g2(g1(concatenate([hypothesis_input, premise_attend])))

        # aggregate
        premise_sum = Lambda(lambda x: K.sum(x, axis=1))(premise_g)
        hypothesis_sum = Lambda(lambda x: K.sum(x, axis=1))(hypothesis_g)

        h1 = Dense(units=200, activation='relu')
        h2 = Dense(units=200, activation='relu')

        premise_hypothesis_h = h2(h1(concatenate([premise_sum, hypothesis_sum])))
        output = Dense(self.config.n_class, activation='softmax')(premise_hypothesis_h)

        model = Model(inputs, output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model

