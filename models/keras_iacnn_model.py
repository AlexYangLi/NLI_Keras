# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_iacnn_model.py

@time: 2019/2/20 11:03

@desc:

"""

from keras.layers import concatenate, subtract, multiply, Conv1D, GlobalMaxPooling1D, \
    BatchNormalization, Dense

from keras import Model

from models.keras_base_model import KerasBaseModel
from layers.attention import DotProductAttention


class KerasIACNNModel(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(KerasIACNNModel, self).__init__(config, **kwargs)

    def build(self, input_config='token', elmo_output_mode='elmo'):
        inputs, premise_embed, hypothesis_embed = self.build_input(input_config=input_config, mask_zero=False,
                                                                   elmo_output_mode=elmo_output_mode)

        premise_attend, hypothesis_attend = DotProductAttention()([premise_embed, hypothesis_embed])
        premise_enhance = concatenate([premise_embed, premise_attend, subtract([premise_embed, premise_attend]),
                                       multiply([premise_embed, premise_attend])])
        hypothesis_enhance = concatenate([hypothesis_embed, hypothesis_attend,
                                          subtract([hypothesis_embed, hypothesis_attend]),
                                          multiply([hypothesis_embed, hypothesis_attend])])

        conv_layers = []
        filter_lengths = [2, 3, 4, 5]
        for filter_length in filter_lengths:
            conv_layer = Conv1D(filters=300, kernel_size=filter_length, padding='valid',
                                strides=1, activation='relu')
            conv_layers.append(conv_layer)

        cnn_premise = concatenate([GlobalMaxPooling1D()(conv_layer(premise_enhance)) for conv_layer in conv_layers])
        cnn_hypothesis = concatenate([GlobalMaxPooling1D()(conv_layer(hypothesis_enhance)) for conv_layer in conv_layers])

        p_concat_h = concatenate([cnn_premise, cnn_hypothesis])

        bn_concat = BatchNormalization()(p_concat_h)
        dense_1 = Dense(self.config.dense_units, activation='relu')(bn_concat)
        bn_dense_1 = BatchNormalization()(dense_1)
        dense_2 = Dense(self.config.dense_units, activation='relu')(bn_dense_1)
        bn_dense_2 = BatchNormalization()(dense_2)

        output = Dense(self.config.n_class, activation='softmax')(bn_dense_2)

        model = Model(inputs, output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model
