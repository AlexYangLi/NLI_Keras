# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_siamese_bilstm_model.py

@time: 2019/2/20 10:06

@desc:

"""

from keras.layers import LSTM, Bidirectional, concatenate, Lambda, BatchNormalization, Dense
from keras import Model, backend as K

from models.keras_base_model import KerasBaseModel


class KerasSimaeseBiLSTMModel(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(KerasSimaeseBiLSTMModel, self).__init__(config, **kwargs)

    def build(self, input_config='token', elmo_output_mode='elmo', elmo_model_url=None, elmo_trainable=None):
        inputs, premise_embed, hypothesis_embed = self.build_input(input_config=input_config, mask_zero=True,
                                                                   elmo_output_mode=elmo_output_mode,
                                                                   elmo_trainable=elmo_trainable,
                                                                   elmo_model_url=elmo_model_url)

        bilstm = Bidirectional(LSTM(units=self.config.rnn_units, return_sequences=True))
        premise_hidden = bilstm(premise_embed)
        hypothesis_hidden = bilstm(hypothesis_embed)

        global_max_pooling = Lambda(lambda x: K.max(x, axis=1))  # GlobalMaxPooling1D didn't support masking
        p_concat_h = concatenate([global_max_pooling(premise_hidden), global_max_pooling(hypothesis_hidden)])

        bn_concat = BatchNormalization()(p_concat_h)
        dense_1 = Dense(self.config.dense_units, activation='relu')(bn_concat)
        bn_dense_1 = BatchNormalization()(dense_1)
        dense_2 = Dense(self.config.dense_units, activation='relu')(bn_dense_1)
        bn_dense_2 = BatchNormalization()(dense_2)

        output = Dense(self.config.n_class, activation='softmax')(bn_dense_2)

        model = Model(inputs, output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model
