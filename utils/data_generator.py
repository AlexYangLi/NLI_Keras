# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: data_generator.py

@time: 2019/3/13 22:07

@desc:

"""

import numpy as np
from keras.utils import Sequence
from utils.data_loader import load_processed_data, load_features
from utils.cache import ELMoCache


class ELMoGenerator(Sequence):
    def __init__(self, genre, level, data_type, batch_size, elmocache: ELMoCache, shuffle=True, return_data=False,
                 return_features=False, return_label=True):
        """
        :param elmocache:  instance of ELMoCache, used to genrate elmo embedding
        """
        self.input_data = load_processed_data(genre, level, data_type)
        self.input_premise = self.input_data['premise']
        self.input_hypothesis = self.input_data['hypothesis']
        self.input_label = self.input_data['label']
        assert self.input_hypothesis.shape[0] == self.input_hypothesis.shape[0] == self.input_label.shape[0]
        self.data_size = self.input_hypothesis.shape[0]
        self.indexes = np.arange(self.data_size)

        self.batch_size = batch_size
        self.elmocache = elmocache
        self.shuffle = shuffle
        self.return_data = return_data      # whether to return original data
        self.return_features = return_features  # whether to return additional statistical features
        self.return_label = return_label    # whether to return label

        if self.return_features:
            self.features = load_features(genre, data_type)

    def __len__(self):
        return int(np.ceil(self.data_size / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_premise = self.input_hypothesis[batch_indexes]
        batch_premise_elmo = self.elmo_generator(batch_premise)
        batch_hypothesis = self.input_hypothesis[batch_indexes]
        batch_hypothesis_elmo = self.elmo_generator(batch_hypothesis)

        if self.return_data:
            batch_data = [batch_premise, batch_hypothesis, batch_premise_elmo, batch_hypothesis_elmo]
        else:
            batch_data = [batch_premise_elmo, batch_hypothesis_elmo]

        if self.return_features:
            batch_features = self.features[batch_indexes]
            batch_data.append(batch_features)

        if self.return_label:
            batch_label = self.input_label[batch_indexes]
            return batch_data, batch_label
        else:
            return batch_data

    def elmo_generator(self, batch_token_ids):
        return self.elmocache.embed_batch(batch_token_ids)







