# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_siamese_lstmcnn_model.py

@time: 2019/4/2 9:13

@desc:

"""

from keras.models import Model
from keras.layers import Dense, Dropout, Bidirectional, LSTM, Conv1D, concatenate, GlobalMaxPooling1D, \
    GlobalAveragePooling1D, BatchNormalization
from keras.regularizers import L1L2
from models.keras_base_model import KerasBaseModel


class KerasSiameseLSTMCNNModel(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(KerasSiameseLSTMCNNModel, self).__init__(config, **kwargs)

    def build(self, input_config='token', elmo_output_mode='elmo', elmo_trainable=None, elmo_model_url=None,
              n_hidden=3):
        inputs, premise_embed, hypothesis_embed = self.build_input(input_config=input_config, mask_zero=False,
                                                                   elmo_output_mode=elmo_output_mode,
                                                                   elmo_trainable=elmo_trainable,
                                                                   elmo_model_url=None)

        premise_hidden = premise_embed
        hypothesis_hidden = hypothesis_embed
        for _ in range(n_hidden):
            lstm = Bidirectional(LSTM(self.config.rnn_units, return_sequences=True,
                                      kernel_regularizer=L1L2(l1=0.01, l2=0.01)))
            premise_hidden = lstm(premise_hidden)
            hypothesis_hidden = lstm(hypothesis_hidden)

        cnn = Conv1D(filters=self.config.rnn_units, kernel_size=3, strides=1)
        premise_cnn = cnn(premise_hidden)
        hypothesis_cnn = cnn(hypothesis_hidden)

        premise_final = concatenate([GlobalMaxPooling1D()(premise_cnn), GlobalAveragePooling1D()(premise_cnn)])
        hypothesis_final = concatenate([GlobalMaxPooling1D()(hypothesis_cnn), GlobalAveragePooling1D()(hypothesis_cnn)])

        if self.config.add_features:
            p_concat_h = concatenate([premise_final, hypothesis_final, inputs[-1]])
        else:
            p_concat_h = concatenate([premise_final, hypothesis_final])

        bn_concat = BatchNormalization()(p_concat_h)
        dense_1 = Dense(self.config.dense_units, activation='relu')(bn_concat)
        bn_dense_1 = BatchNormalization()(dense_1)
        dropout_1 = Dropout(self.config.dropout)(bn_dense_1)
        dense_2 = Dense(self.config.dense_units, activation='relu')(dropout_1)
        bn_dense_2 = BatchNormalization()(dense_2)
        dropout_2 = Dropout(self.config.dropout)(bn_dense_2)

        output = Dense(self.config.n_class, activation='softmax')(dropout_2)

        model = Model(inputs, output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model
