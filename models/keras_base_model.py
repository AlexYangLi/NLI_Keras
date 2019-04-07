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
import math

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Embedding, concatenate
from layers.embedding import ELMoEmbedding
from models.base_model import BaseModel
from layers.weight_average import WeightedAverage
from callbacks.callbacks import LRRangeTest, CyclicLR, SWA, SWAWithCyclicLR
from utils.metrics import eval_acc


class KerasBaseModel(BaseModel):
    def __init__(self, config, **kwargs):
        super(KerasBaseModel, self).__init__()
        self.config = config
        self.level = self.config.input_level
        self.max_len = self.config.max_len
        self.word_embeddings = config.word_embeddings

        self.callbacks = []
        # self.init_callbacks()

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

    def add_model_checkpoint(self):
        self.callbacks.append(ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_dir, '{}.hdf5'.format(self.config.exp_name)),
            monitor=self.config.checkpoint_monitor,
            save_best_only=self.config.checkpoint_save_best_only,
            save_weights_only=self.config.checkpoint_save_weights_only,
            mode=self.config.checkpoint_save_weights_mode,
            verbose=self.config.checkpoint_verbose
        ))

    def add_early_stopping(self):
        self.callbacks.append(EarlyStopping(
            monitor=self.config.early_stopping_monitor,
            mode=self.config.early_stopping_mode,
            patience=self.config.early_stopping_patience,
            verbose=self.config.early_stopping_verbose
        ))

    def add_swa(self, swa_start=5):
        self.callbacks.append(SWA(self.config.checkpoint_dir, self.config.exp_name, swa_start=swa_start))

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def load_model(self, filename):
        # we only save model's weight instead of the whole model
        self.model.load_weights(filename)

    def load_best_model(self):
        print('Logging Info - Loading model checkpoint: %s.hdf5\n' % self.config.exp_name)
        self.load_model(os.path.join(self.config.checkpoint_dir, '{}.hdf5'.format(self.config.exp_name)))
        print('Logging Info - Model loaded')

    def load_swa_model(self):
        print('Logging Info - Loading SWA model checkpoint: %s_swa.hdf5\n' % self.config.exp_name)
        self.load_model(os.path.join(self.config.checkpoint_dir, '{}_swa.hdf5'.format(self.config.exp_name)))
        print('Logging Info - SWA Model loaded')

    @abc.abstractmethod
    def build(self, **kwargs):
        """Build Keras Model, specified by different models"""

    def train(self, x_train, y_train, x_valid, y_valid):
        self.callbacks = []
        self.add_model_checkpoint()
        self.add_swa()

        if self.config.use_cyclical_lr:
            '''
            when use cyclical lr, step_size is suggested to be 2-8 x training iterations in epoch,
            also num_epoch is suggested to be multiple of 2*step_size(cycle)
            '''
            step_size = 2 * math.floor(y_train.shape[0] / self.config.batch_size)
            self.callbacks.append(CyclicLR(base_lr=self.config.base_lr, max_lr=self.config.max_lr,
                                           step_size=step_size, mode='triangular2'))
            # self.callbacks.append(SWAWithCyclicLR(self.config.checkpoint_dir, self.config.exp_name, swa_start=1,
            #                                       base_lr=self.config.base_lr, max_lr=self.config.max_lr,
            #                                       step_size=step_size, mode='triangular2'))
        else:
            self.add_early_stopping()   # when using cyclical lr, we don't use early stopping

        print('Logging Info - Start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size, epochs=self.config.n_epoch,
                       validation_data=(x_valid, y_valid), callbacks=self.callbacks)
        print('Logging Info - Training end...')

    def train_with_generator(self, train_generator, valid_generator):
        self.callbacks = []
        self.init_callbacks()

        print('Logging Info - Start training...')
        self.model.fit_generator(generator=train_generator, epochs=self.config.n_epoch, callbacks=self.callbacks,
                                 validation_data=valid_generator)
        print('Logging Info - Training end...')

    def predict(self, x):
        return self.model.predict(x)

    def predict_with_generator(self, generator):
        return self.model.predict_generator(generator=generator)

    def evaluate(self, x, y):
        prediction = self.predict(x)
        acc = eval_acc(y, prediction)
        print('Logging Info - Acc : %f' % acc)
        return acc

    def evaluate_with_generator(self, generator, y):
        prediction = self.predict_with_generator(generator)
        acc = eval_acc(y, prediction)
        print('Logging Info - Acc : %f' % acc)
        return acc

    def lr_range_test(self, x_train, y_train, x_valid, y_valid):
        # conduct `lr range test` experiment to find the optimal learning rate range
        self.callbacks = []
        if self.config.use_cyclical_lr:
            step_size = self.config.n_epoch * math.floor(y_train.shape[0] / self.config.batch_size)
            self.callbacks.append(CyclicLR(base_lr=1e-5, max_lr=1., step_size=step_size, mode='triangular',
                                           plot=True, save_plot_prefix=self.config.exp_name+'_cyclic'))
        else:
            num_batches = self.config.n_epoch * math.floor(y_train.shape[0] / self.config.batch_size)
            self.callbacks.append(LRRangeTest(num_batches, save_plot_prefix=self.config.exp_name))

        print('Logging Info - Start LR Range Test...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size, epochs=self.config.n_epoch,
                       validation_data=(x_valid, y_valid), callbacks=self.callbacks)
        print('Logging Info -  LR Range Test end...')

    def summary(self):
        self.model.summary()

    def build_input(self, input_config='token', mask_zero=True, elmo_model_url=None,
                    elmo_output_mode='elmo', elmo_trainable=None):
        """
        build input embeddings layer, same across different models, so we implement here.
        :param input_config: determine the model input, options are:
                             1. 'token', using token ids as input and one embedding layer to lookup their embeddings
                             2. 'elmo_id', using token ids as input and ELMoEmbedding layer (based on tfhub) to generate
                                elmo embeddings. `elmo_output_mode` determine which elmo output to use, `elmo_trainable`
                                determine whether ELMoEmbedding layer is trainable. See layers.embedding.py for more
                                details.
                             3. `elmo_s', using untokenized sentences and ELMoEmbedding layer (based on tfhub) to
                                generate elmo embeddings. See layers.embedding.py for more details.
                             4. 'cache_elmo', using cache elmo embedding (fed with fit_generator) as input
                             5. 'token_combine_elmo_id', combining the embedding outputs from 1 and 2
                             6. 'token_combine_elmo_s', combining the embedding outputs from 1 and 3
                             7. 'token_combine_elmo', combining the embedding outputs from 1 and 6
        :param mask_zero: whether to apply masking
        :param elmo_output_mode: determine which elmo output to use, options are:
                                 1. 'word_embed', using context-independent word embedding
                                 2. 'lstm_outputs1', using first lstm layer output
                                 3. 'lstm_outputs2', using second lstm layer output
                                 4. 'elmo', using weighted sum of three layers' outputs
                                 5. 'elmo_avg', using averge of three layers' outputs
                                 6. 'default', using sentence embedding as output, in this situation, elmo embedding
                                    can not be combined with token embeddings
        :param elmo_trainable: determine whether ELMoEmbedding layer is trainable
        """
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

            embedding = ELMoEmbedding(output_mode=elmo_output_mode, idx2word=self.config.idx2token, mask_zero=mask_zero,
                                      hub_url=elmo_model_url, elmo_trainable=elmo_trainable)
            premise_embed = embedding(input_premise)
            hypothesis_embed = embedding(input_hypothesis)
        elif input_config == 'elmo_s':
            # use untokenized sentences to get elmo embeddings
            input_premise = Input(shape=(1, ), dtype='string')
            input_hypothesis = Input(shape=(1, ), dtype='string')
            inputs = [input_premise, input_hypothesis]

            embedding = ELMoEmbedding(output_mode=elmo_output_mode, max_length=self.max_len, mask_zero=mask_zero,
                                      hub_url=elmo_model_url, elmo_trainable=elmo_trainable)
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

            elmo_embedding = ELMoEmbedding(output_mode=elmo_output_mode, idx2word=self.config.idx2token,
                                           mask_zero=mask_zero, hub_url=elmo_model_url, elmo_trainable=elmo_trainable)

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
            elmo_embedding = ELMoEmbedding(output_mode=elmo_output_mode, max_length=self.max_len, mask_zero=mask_zero,
                                           hub_url=elmo_model_url, elmo_trainable=elmo_trainable)
            premise_embed = concatenate([token_embedding(input_premise_id), elmo_embedding(input_premise_s)])
            hypothesis_embed = concatenate([token_embedding(input_hypothesis_id), elmo_embedding(input_hypothesis_s)])
        elif input_config == 'cache_elmo':
            if elmo_output_mode == 'elmo':
                weight_layer = WeightedAverage()
                input_premise = Input(shape=(3, self.max_len, 1024))
                input_hypothesis = Input(shape=(3, self.max_len, 1024))
                inputs = [input_premise, input_hypothesis]

                premise_embed = weight_layer(input_premise)
                hypothesis_embed = weight_layer(input_hypothesis)
            else:
                input_premise = Input(shape=(self.max_len, 1024))
                input_hypothesis = Input(shape=(self.max_len, 1024))
                inputs = [input_premise, input_hypothesis]

                premise_embed = input_premise
                hypothesis_embed = input_hypothesis
        elif input_config == 'token_combine_cache_elmo':
            input_premise_id = Input(shape=(self.max_len,))
            input_hypothesis_id = Input(shape=(self.max_len,))
            token_embedding = Embedding(self.word_embeddings.shape[0], self.word_embeddings.shape[1],
                                        weights=[self.word_embeddings], trainable=self.config.word_embed_trainable,
                                        mask_zero=mask_zero)
            inputs = [input_premise_id, input_hypothesis_id]
            if elmo_output_mode == 'elmo':
                weight_layer = WeightedAverage()
                input_premise_cache = Input(shape=(3, self.max_len, 1024))
                input_hypothesis_cache = Input(shape=(3, self.max_len, 1024))
                inputs.extend([input_premise_cache, input_premise_cache])

                premise_embed = concatenate([token_embedding(input_premise_id), weight_layer(input_premise_cache)])
                hypothesis_embed = concatenate([token_embedding(input_hypothesis_id),
                                                weight_layer(input_hypothesis_cache)])
            else:
                input_premise_cache = Input(shape=(self.max_len, 1024))
                input_hypothesis_cache = Input(shape=(self.max_len, 1024))
                inputs.extend([input_premise_cache, input_premise_cache])

                premise_embed = concatenate([token_embedding(input_premise_id), input_premise_cache])
                hypothesis_embed = concatenate([token_embedding(input_hypothesis_id), input_hypothesis_cache])
        else:
            raise ValueError('input_config Not Understood:{}'.format(input_config))

        if self.config.add_features:
            inputs.append(Input(shape=(self.config.feature_len, )))
        return inputs, premise_embed, hypothesis_embed
