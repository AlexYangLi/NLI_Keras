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

from keras import optimizers
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from models.keras_infersent_model import KerasInfersentModel
from models.keras_esim_model import KerasEsimModel
from models.keras_decomposable_model import KerasDecomposableAttentionModel
from config import ModelConfig, PERFORMANCE_LOG, LOG_DIR
from utils.data_loader import load_processed_data
from utils.io import write_log, format_filename

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ex = Experiment('train_model')
# ex.observers.append(FileStorageObserver.create(LOG_DIR))
# ex.captured_out_filter = apply_backspaces_and_linefeeds


def get_optimizer(op_type, learning_rate):
    if op_type == 'sgd':
        return optimizers.SGD(learning_rate)
    elif op_type == 'rmsprop':
        return optimizers.RMSprop(learning_rate)
    elif op_type == 'adagrad':
        return optimizers.Adagrad(learning_rate)
    elif op_type == 'adadelta':
        return optimizers.Adadelta(learning_rate)
    elif op_type == 'adam':
        return optimizers.Adam(learning_rate)
    else:
        raise ValueError('Optimizer Not Understood: {}'.format(op_type))


@ex.config
def param_config():
    genre = 'snli'
    input_level = 'word'
    word_embed_type = 'glove_cc'
    word_embed_trainable = False
    batch_size = 512
    learning_rate = 0.001
    optimizer_type = 'adam'
    model_name = 'KerasDecomposable'
    exp_name = '{}_{}_{}_{}_{}'.format(genre, model_name, input_level, word_embed_type,
                                       'tune' if word_embed_trainable else 'fix')


@ex.main
def train_model(genre, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate, optimizer_type,
                model_name, exp_name):
    config = ModelConfig()
    config.genre = genre
    config.input_level = input_level
    config.word_embed_type = word_embed_type
    config.word_embed_trainable = word_embed_trainable
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.optimizer = get_optimizer(optimizer_type, learning_rate)
    config.exp_name = exp_name

    train_log = {'exp_name': exp_name, 'batch_size': batch_size, 'optimizer': optimizer_type,
                 'learning_rate': learning_rate}

    logging.info('Experiment: %s', exp_name)
    if model_name == 'KerasInfersent':
        model = KerasInfersentModel(config)
    elif model_name == 'KerasEsim':
        model = KerasEsimModel(config)
    elif model_name == 'KerasDecomposable':
        model = KerasDecomposableAttentionModel(config)
    else:
        raise ValueError('Model Name Not Understood : {}'.format(model_name))

    train_input = load_processed_data(genre, input_level, 'train')
    dev_input = load_processed_data(genre, input_level, 'dev')
    test_input = load_processed_data(genre, input_level, 'test')

    model_save_path = config.checkpoint_dir / '{}.hdf5'.format(config.exp_name)
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
    train_log['train_acc'] = train_acc

    logging.info('evaluate over valid data:')
    valid_acc = model.evaluate(dev_input)
    train_log['valid_acc'] = valid_acc

    logging.info('evaluate over test data...')
    test_acc = model.evaluate(test_input)
    train_log['test_acc'] = test_acc

    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG), log=train_log, mode='a')
    return train_log


if __name__ == '__main__':
    ex.run_commandline()
