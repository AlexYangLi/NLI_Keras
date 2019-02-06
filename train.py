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
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from models.keras_infersent_model import KerasInfersentModel
from config import ModelConfig, PERFORMANCE_LOG, LOG_DIR
from utils.data_loader import load_processed_data
from utils.io import write_log, format_filename

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
ex = Experiment('train_model')
ex.observers.append(FileStorageObserver.create(LOG_DIR))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def param_config():
    genre = 'snli'
    model_name = 'KerasInfersent'
    input_level = 'word'
    word_embed_type = 'glove_cc'
    word_embed_trainable = False
    exp_name = '{}_{}_{}_{}_{}'.format(genre, model_name, input_level, word_embed_type,
                                       'tune' if word_embed_trainable else 'fix')


@ex.main
def train_model(genre, input_level, word_embed_type, word_embed_trainable, model_name, exp_name):
    train_log = dict()
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

    train_input = load_processed_data(genre, input_level, 'train')
    dev_input = load_processed_data(genre, input_level, 'dev')
    test_input = load_processed_data(genre, input_level, 'test')

    model_save_path = config.checkpoint_dir / '{}_{}.hdf5'.format(config.genre, config.exp_name)
    if not model_save_path.exists():
        start_time = time.time()
        model.train(train_input, dev_input)
        elapsed_time = time.time() - start_time
        logging.info('training time: %s', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        train_log[exp_name+'_train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    # load the best model
    model.load_best_model()

    logging.info('evaluate over train data:')
    train_acc = model.evaluate(train_input)
    train_log[exp_name + '_train_acc'] = train_acc

    logging.info('evaluate over valid data:')
    valid_acc = model.evaluate(dev_input)
    train_log[exp_name+'_valid_acc'] = valid_acc

    logging.info('evaluate over test data...')
    test_acc = model.evaluate(test_input)
    train_log[exp_name+'_test_acc'] = test_acc

    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG), log=train_log, mode='a')
    return train_log


if __name__ == '__main__':
    ex.run_commandline()
