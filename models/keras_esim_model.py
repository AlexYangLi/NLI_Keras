# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_esim_model.py

@time: 2019/2/3 17:15

@desc: enhanced sequential inference model(esim) without tree-based component

"""

from keras.layers import Input, Embedding, LSTM, Bidirectional, GlobalAveragePooling1D, \
      concatenate, Lambda, subtract, multiply, Dense, TimeDistributed
import keras.backend as K
from keras import Model

from models.keras_base_model import KerasBaseModel
from layers.attention import DotProductAttention


class KerasEsimModel(KerasBaseModel):
    def __init__(self, config):
        super(KerasEsimModel, self).__init__(config)

    def build(self):
        input_premise = Input(shape=(self.max_len,))
        input_hypothesis = Input(shape=(self.max_len,))

        embedding = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                              weights=[self.word_embeddings], trainable=self.config.word_embed_trainable,
                              mask_zero=True)
        premise_embed = embedding(input_premise)
        hypothesis_embed = embedding(input_hypothesis)

        # input encoding
        bilstm_1 = Bidirectional(LSTM(units=300, return_sequences=True))
        premise_hidden = bilstm_1(premise_embed)
        hypothesis_hidden = bilstm_1(hypothesis_embed)

        # local inference collected over sequences
        premise_attend, hypothesis_attend = DotProductAttention()([premise_hidden, hypothesis_hidden])

        # enhancement of local inference information
        premise_enhance = concatenate([premise_hidden, premise_attend, subtract([premise_hidden, premise_attend]),
                                       multiply([premise_hidden, premise_attend])])
        hypothesis_enhance = concatenate([hypothesis_hidden, hypothesis_attend,
                                          subtract([hypothesis_hidden, hypothesis_attend]),
                                          multiply([hypothesis_hidden, hypothesis_attend])])

        # inference composition
        feed_forward = TimeDistributed(Dense(units=300, activation='relu'))
        bilstm_2 = Bidirectional(LSTM(units=300, return_sequences=True))
        premise_compose = bilstm_2(feed_forward(premise_enhance))
        hypothesis_compose = bilstm_2(feed_forward(hypothesis_enhance))

        global_max_pooling = Lambda(lambda x: K.max(x, axis=1))  # GlobalMaxPooling1D didn't support masking
        premise_avg = GlobalAveragePooling1D()(premise_compose)
        premise_max = global_max_pooling(premise_compose)
        hypothesis_avg = GlobalAveragePooling1D()(hypothesis_compose)
        hypothesis_max = global_max_pooling(hypothesis_compose)

        inference_compose = concatenate([premise_avg, premise_max, hypothesis_avg, hypothesis_max])
        dense = Dense(units=300, activation='tanh')(inference_compose)
        output = Dense(self.config.n_class, activation='softmax')(dense)

        model = Model([input_premise, input_hypothesis], output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model




