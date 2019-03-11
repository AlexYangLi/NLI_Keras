# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_base_model.py

@time: 2019/2/3 17:14

@desc:

"""

import os
import abc

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Embedding, concatenate
from layers.embedding import ELMoEmbedding
from models.base_model import BaseModel
from utils.metrics import eval_acc


class KerasBaseModel(BaseModel):
    def __init__(self, config, **kwargs):
        super(KerasBaseModel, self).__init__()
        self.config = config
        self.level = self.config.input_level
        self.max_len = self.config.max_len
        self.word_embeddings = config.word_embeddings

        self.callbacks = []
        self.init_callbacks()

        self.model = self.build(**kwargs)

    def init_callbacks(self):
        self.callbacks.append(ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_dir, '{}.hdf5'.format(self.config.exp_name)),
            monitor=self.config.checkpoint_monitor,
            save_best_only=self.config.checkpoint_save_best_only,
            save_weights_only=self.config.checkpoint_save_weights_only,
            mode=self.config.checkpoint_save_weights_mode,
            verbose=self.config.checkpoint_verbose
        ))

        self.callbacks.append(EarlyStopping(
            monitor=self.config.early_stopping_monitor,
            mode=self.config.early_stopping_mode,
            patience=self.config.early_stopping_patience,
            verbose=self.config.early_stopping_verbose
        ))

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def load_model(self, filename):
        # we only save model's weight instead of the whole model
        self.model.load_weights(filename)

    def load_best_model(self):
        print('Logging Info - Loading model checkpoint: %s.hdf5\n' % self.config.exp_name)
        self.load_model(os.path.join(self.config.checkpoint_dir, '{}.hdf5'.format(self.config.exp_name)))
        print('Logging Info - Model loaded')

    @abc.abstractmethod
    def build(self, **kwargs):
        """Build Keras Model, specified by different models"""

    def train(self, x_train, y_train, x_valid, y_valid):
        print('Logging Info - Start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size, epochs=self.config.n_epoch,
                       validation_data=(x_valid, y_valid), callbacks=self.callbacks)
        print('Logging Info - Training end...')

    def evaluate(self, x, y):
        prediction = self.predict(x)
        acc = eval_acc(y, prediction)
        print('Logging Info - Acc : %f' % acc)
        return acc

    def predict(self, x):
        return self.model.predict(x)

    def build_input(self, input_config='token', mask_zero=True, elmo_output_mode='elmo'):
        """build input embeddings layer, same across different models, so we implement here"""
        if input_config == 'token':
            # only use token(word or character level) embedding
            input_premise = Input(shape=(self.max_len,))
            input_hypothesis = Input(shape=(self.max_len,))
            inputs = [input_premise, input_hypothesis]

            embedding = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                  weights=[self.word_embeddings], trainable=self.config.word_embed_trainable,
                                  mask_zero=mask_zero)
            premise_embed = embedding(input_premise)
            hypothesis_embed = embedding(input_hypothesis)
        elif input_config == 'elmo_id':
            # use token id sequence to get elmo embeddings
            input_premise = Input(shape=(self.max_len,))
            input_hypothesis = Input(shape=(self.max_len,))
            inputs = [input_premise, input_hypothesis]

            from config import EXTERNAL_WORD_VECTORS_FILENAME
            path_to_elmo_model = EXTERNAL_WORD_VECTORS_FILENAME['tfhub_elmo_2']
            embedding = ELMoEmbedding(output_mode=elmo_output_mode, idx2word=self.config.idx2token, mask_zero=mask_zero,
                                      hub_url=path_to_elmo_model)
            premise_embed = embedding(input_premise)
            hypothesis_embed = embedding(input_hypothesis)
        elif input_config == 'elmo_s':
            # use untokenized sentences to get elmo embeddings
            input_premise = Input(shape=(1, ), dtype='string')
            input_hypothesis = Input(shape=(1, ), dtype='string')
            inputs = [input_premise, input_hypothesis]

            from config import EXTERNAL_WORD_VECTORS_FILENAME
            path_to_elmo_model = EXTERNAL_WORD_VECTORS_FILENAME['tfhub_elmo_2']
            embedding = ELMoEmbedding(output_mode=elmo_output_mode, max_length=self.max_len, mask_zero=mask_zero,
                                      hub_url=path_to_elmo_model)
            premise_embed = embedding(input_premise)
            hypothesis_embed = embedding(input_hypothesis)
        elif input_config == 'token_combine_elmo_id':
            # use token embedding and elmo embedding
            input_premise = Input(shape=(self.max_len,))
            input_hypothesis = Input(shape=(self.max_len,))
            inputs = [input_premise, input_hypothesis]

            token_embedding = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                        weights=[self.word_embeddings], trainable=self.config.word_embed_trainable,
                                        mask_zero=mask_zero)

            from config import EXTERNAL_WORD_VECTORS_FILENAME
            path_to_elmo_model = EXTERNAL_WORD_VECTORS_FILENAME['tfhub_elmo_2']
            elmo_embedding = ELMoEmbedding(output_mode=elmo_output_mode, idx2word=self.config.idx2token,
                                           mask_zero=mask_zero, hub_url=path_to_elmo_model)

            premise_embed = concatenate([token_embedding(input_premise), elmo_embedding(input_premise)])
            hypothesis_embed = concatenate([token_embedding(input_hypothesis), elmo_embedding(input_hypothesis)])
        elif input_config == 'token_combine_elmo_s':
            # use token embedding and elmo embedding
            input_premise_id = Input(shape=(self.max_len,))
            input_hypothesis_id = Input(shape=(self.max_len,))
            input_premise_s = Input(shape=(1,), dtype='string')
            input_hypothesis_s = Input(shape=(1,), dtype='string')
            inputs = [input_premise_id, input_hypothesis_id, input_premise_s, input_hypothesis_s]

            token_embedding = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                        weights=[self.word_embeddings], trainable=self.config.word_embed_trainable,
                                        mask_zero=mask_zero)
            from config import EXTERNAL_WORD_VECTORS_FILENAME
            path_to_elmo_model = EXTERNAL_WORD_VECTORS_FILENAME['tfhub_elmo_2']
            elmo_embedding = ELMoEmbedding(output_mode=elmo_output_mode, max_length=self.max_len, mask_zero=mask_zero,
                                           hub_url=path_to_elmo_model)
            premise_embed = concatenate([token_embedding(input_premise_id), elmo_embedding(input_premise_s)])
            hypothesis_embed = concatenate([token_embedding(input_hypothesis_id), elmo_embedding(input_hypothesis_s)])
        else:
            raise ValueError('input_config Not Understood:{}'.format(input_config))
        return inputs, premise_embed, hypothesis_embed
