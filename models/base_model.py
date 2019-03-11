# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: base_model.py

@time: 2019/2/1 14:03

@desc:

"""

import abc


class BaseModel(object):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.model = None

    @abc.abstractmethod
    def build(self):
        """Build the model"""

    @abc.abstractmethod
    def train(self, x_train, y_train, x_valid, y_valid):
        """Train the model"""

    @abc.abstractmethod
    def load_weights(self, filename):
        """Load weights from the `filename`"""

    @abc.abstractmethod
    def load_model(self, filename):
        """Load models from the `filename`"""

    @abc.abstractmethod
    def evaluate(self, x, y):
        """Evaluate the model on the provided data"""

    @abc.abstractmethod
    def predict(self, x):
        """Predict for the provided data"""
