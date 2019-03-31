# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_infersent_model.py

@time: 2019/2/3 17:13

@desc:

"""
from keras.layers import LSTM, GRU, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, \
      concatenate, Conv1D, Lambda, multiply, Dense
import keras.backend as K
from keras import Model

from models.keras_base_model import KerasBaseModel
from layers.attention import SelfAttention


class KerasInfersentModel(KerasBaseModel):
    def __init__(self, config, **kwargs):
        super(KerasInfersentModel, self).__init__(config, **kwargs)

    def build(self, input_config='token', elmo_output_mode='elmo', elmo_trainable=None, elmo_model_url=None,
              encoder_type='bilstm_max_pool'):
        mask_zero = False if encoder_type in ['h_cnn'] else True    # cnn doesn't support masking

        inputs, premise_embed, hypothesis_embed = self.build_input(input_config=input_config, mask_zero=mask_zero,
                                                                   elmo_output_mode=elmo_output_mode,
                                                                   elmo_trainable=elmo_trainable,
                                                                   elmo_model_url=elmo_model_url)

        premise_encoded, hypothesis_encoded = self.sentence_encoder(premise_embed, hypothesis_embed, encoder_type)
        p_sub_h = Lambda(lambda x: K.abs(x[0] - x[1]))([premise_encoded, hypothesis_encoded])
        p_mul_h = multiply([premise_encoded, hypothesis_encoded])

        if self.config.add_features:
            p_cancat_h = concatenate([premise_encoded, hypothesis_encoded, p_sub_h, p_mul_h, inputs[-1]])
        else:
            p_cancat_h = concatenate([premise_encoded, hypothesis_encoded, p_sub_h, p_mul_h])

        dense = Dense(units=self.config.dense_units, activation='relu')(p_cancat_h)
        output = Dense(self.config.n_class, activation='softmax')(dense)

        model = Model(inputs, output)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)
        return model

    def sentence_encoder(self, premise, hypothesis, encoder_type):
        if encoder_type == 'lstm':
            lstm = LSTM(units=self.config.rnn_units)
            return lstm(premise), lstm(hypothesis)
        elif encoder_type == 'gru':
            gru = GRU(units=self.config.rnn_units)
            return gru(premise), gru(hypothesis)
        elif encoder_type == 'bilstm':
            bilstm = Bidirectional(LSTM(units=self.config.rnn_units))
            return bilstm(premise), bilstm(hypothesis)
        elif encoder_type == 'bigru':
            bigru = Bidirectional(GRU(units=self.config.rnn_units))
            return bigru(premise), bigru(hypothesis)
        elif encoder_type == 'bilstm_max_pool':
            bilstm = Bidirectional(LSTM(units=self.config.rnn_units, return_sequences=True))
            global_max_pooling = Lambda(lambda x: K.max(x, axis=1))     # GlobalMaxPooling1D didn't support masking
            return global_max_pooling(bilstm(premise)), global_max_pooling(bilstm(hypothesis))
        elif encoder_type == 'bilstm_mean_pool':
            bilstm = Bidirectional(LSTM(units=self.config.rnn_units, return_sequences=True))
            return GlobalAveragePooling1D()(bilstm(premise)), GlobalAveragePooling1D()(bilstm(hypothesis))
        elif encoder_type == 'self_attentive':
            attention_layers = [SelfAttention() for _ in range(4)]
            attend_premise = [attend_layer(premise) for attend_layer in attention_layers]
            attend_hypothesis = [attend_layer(hypothesis) for attend_layer in attention_layers]
            return concatenate(attend_premise), concatenate(attend_hypothesis)
        elif encoder_type == 'h_cnn':
            cnn_premise, cnn_hypothesis = [premise], [hypothesis]

            filter_lengths = [2, 3, 4, 5]
            for filter_length in filter_lengths:
                conv_layer = Conv1D(filters=300, kernel_size=filter_length, padding='valid',
                                    strides=1, activation='relu')
                cnn_premise.append(conv_layer(cnn_premise[-1]))
                cnn_hypothesis.append(conv_layer(cnn_hypothesis[-1]))

            global_max_pooling = Lambda(lambda x: K.max(x, axis=1))  # GlobalMaxPooling1D didn't support masking
            cnn_premise = [global_max_pooling(cnn_premise[i]) for i in range(1, 5)]
            cnn_hypothesis = [global_max_pooling(cnn_hypothesis[i]) for i in range(1, 5)]
            return concatenate(cnn_premise), concatenate(cnn_hypothesis)
        else:
            raise ValueError('Encoder Type Not Understood : {}'.format(encoder_type))

