# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_siamese_bilstm_model.py

@time: 2019/2/20 10:06

@desc:

"""

from keras.layers import Input, Embedding, LSTM, Bidirectional, concatenate, Lambda, BatchNormalization, Dense
from keras import Model, backend as K

from models.keras_base_model import KerasBaseModel


class KerasSimaeseBiLSTMModel(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(KerasSimaeseBiLSTMModel, self).__init__(config, **kwargs)

    def build(self):
        input_premise = Input(shape=(self.max_len,))
        input_hypothesis = Input(shape=(self.max_len,))

        embedding = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                              weights=[self.word_embeddings], trainable=self.config.word_embed_trainable,
                              mask_zero=True)
        premise_embed = embedding(input_premise)
        hypothesis_embed = embedding(input_hypothesis)

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

        model = Model([input_premise, input_hypothesis], output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model
