# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: keras_base_model.py

@time: 2019/2/3 17:14

@desc:

"""

import abc
import logging
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping
from config import EMBEDDING_MATRIX_TEMPLATE
from models.base_model import BaseModel
from utils.metrics import eval_acc


class KerasBaseModel(BaseModel):
    def __init__(self, config):
        super(KerasBaseModel, self).__init__()
        self.config = config
        self.level = self.config.level
        self.max_len = self.config.word_max_len if self.level == 'word' else self.config.char_max_len
        self.word_embeddings = np.load(EMBEDDING_MATRIX_TEMPLATE.format(self.config.level,
                                                                        self.config.word_embed_type))

        self.callbacks = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(ModelCheckpoint(
            filepath=self.config.checkpoint_dir / '%s.hdf5' % self.config.exp_name,
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
        logging.info('loading model checkpoint: %s.hdf5\n' % self.config.exp_name)
        self.model.load_weights(self.config.checkpoint_dir / '%s_%s.hdf5' % (self.config.genre, self.config.exp_name))
        logging.info('Model loaded')

    @abc.abstractmethod
    def build(self):
        """Build the model"""

    def train(self, data_train, data_dev):
        if self.model is None:
            self.model = self.build()
        self.model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)

        x_train = [data_train['premise'], data_train['hypothesis']]
        y_train = data_train['label']
        x_valid = [data_dev['premise'], data_dev['hypothesis']]
        y_valid = data_dev['label']

        logging.info('start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size, epochs=self.config.n_epoch,
                       validation_data=(x_valid, y_valid), callbacks=self.callbacks)
        logging.info('training end...')

        logging.info('evaluate over valid data:')
        self.evaluate(data_dev)

    def evaluate(self, data):
        prediction = self.predict(data)
        acc = eval_acc(data['label'], prediction)
        logging.info('acc : %f', acc)

    def predict(self, data):
        return self.model.predict([data['premise'], data['hypothesis']])
