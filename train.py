# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: train.py

@time: 2019/2/1 14:06

@desc:

"""

import os
import time
import logging
from sacred import Experiment
from models.keras_infersent_model import KerasInfersentModel
from config import ModelConfig
from utils.data_loader import load_processed_data

os.environ['CUDA_VSIBLE_DEVICES'] = '1'
ex = Experiment('train_model')


@ex.config
def param_config():
    genre = 'snil'
    model_name = 'KerasInfersent'
    input_level = 'word'
    word_embed_type = 'glove_cc'
    word_embed_trainable = False
    exp_name = '{}_{}_{}_{}_{}'.format(genre, model_name, input_level, word_embed_type,
                                       'tune' if word_embed_trainable else 'fix')


@ex.main
def train_model(genre, input_level, word_embed_type, word_embed_trainable, model_name, exp_name):
    config = ModelConfig()
    config.genre = genre
    config.input_level = input_level
    config.word_embed_type = word_embed_type
    config.word_embed_trainable = word_embed_trainable
    config.exp_name = exp_name

    logging.info('Experiment: %s', exp_name)
    if model_name == 'KerasInfersent':
        model = KerasInfersentModel(config)
    else:
        raise ValueError('Model Name Not Understood : {}'.format(model_name))

    dev_input = load_processed_data(genre, input_level, 'dev')
    test_input = load_processed_data(genre, input_level, 'test')

    model_save_path = config.checkpoint_dir / '%s.hdf5' % self.config.exp_name
    if not model_save_path.exists():
        start_time = time.time()

        train_input = load_processed_data(genre, input_level, 'train')
        model.train(train_input, dev_input)

        elapsed_time = time.time() - start_time
        logging.info('training time: %s', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # load the best model
    model.load_weights()
    logging.info('evaluate over test data...')
    model.evaluate(test_input)


if __name__ == '__main__':
    ex.run_commandline()