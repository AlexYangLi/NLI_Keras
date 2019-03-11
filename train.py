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
import numpy as np
from itertools import product
from keras import optimizers

from models.keras_infersent_model import KerasInfersentModel
from models.keras_esim_model import KerasEsimModel
from models.keras_decomposable_model import KerasDecomposableAttentionModel
from models.keras_siamese_bilstm_model import KerasSimaeseBiLSTMModel
from models.keras_siamese_cnn_model import KerasSiameseCNNModel
from models.keras_iacnn_model import KerasIACNNModel
from config import ModelConfig, PERFORMANCE_LOG, LOG_DIR, PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, \
    VOCABULARY_TEMPLATE
from utils.data_loader import load_input_data
from utils.io import write_log, format_filename, pickle_load

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


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


def train_model(genre, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate, optimizer_type,
                model_name, overwrite=False, eval_on_train=False, **kwargs):
    config = ModelConfig()
    config.genre = genre
    config.input_level = input_level
    config.max_len = config.word_max_len[genre] if input_level == 'word' else config.char_max_len[genre]
    config.word_embed_type = word_embed_type
    config.word_embed_trainable = word_embed_trainable
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.optimizer = get_optimizer(optimizer_type, learning_rate)
    config.word_embeddings = np.load(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, genre,
                                                     word_embed_type))
    vocab = pickle_load(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, genre, input_level))
    config.idx2token = dict((idx, token) for token, idx in vocab.items())

    config.exp_name = '{}_{}_{}_{}_{}_{}_{}'.format(genre, model_name, input_level, word_embed_type,
                                                    'tune' if word_embed_trainable else 'fix', batch_size,
                                                    '_'.join([str(k)+'_'+str(v) for k, v in kwargs.items()]))

    train_log = {'exp_name': config.exp_name, 'batch_size': batch_size, 'optimizer': optimizer_type,
                 'learning_rate': learning_rate, 'other_params': kwargs}

    print('Logging Info - Experiment: %s' % config.exp_name)
    if model_name == 'KerasInfersent':
        model = KerasInfersentModel(config, **kwargs)
    elif model_name == 'KerasEsim':
        model = KerasEsimModel(config, **kwargs)
    elif model_name == 'KerasDecomposable':
        model = KerasDecomposableAttentionModel(config, **kwargs)
    elif model_name == 'KerasSiameseBiLSTM':
        model = KerasSimaeseBiLSTMModel(config, **kwargs)
    elif model_name == 'KerasSiameseCNN':
        model = KerasSiameseCNNModel(config, **kwargs)
    elif model_name == 'KerasIACNN':
        model = KerasIACNNModel(config, **kwargs)
    else:
        raise ValueError('Model Name Not Understood : {}'.format(model_name))

    input_config = kwargs['input_config'] if 'input_config' in kwargs else 'token'
    train_input = load_input_data(genre, input_level, 'train', input_config)
    dev_input = load_input_data(genre, input_level, 'dev', input_config)
    test_input = load_input_data(genre, input_level, 'test', input_config)

    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    if not os.path.exists(model_save_path) or overwrite:
        start_time = time.time()
        model.train(x_train=train_input['x'], y_train=train_input['y'], x_valid=dev_input['x'], y_valid=dev_input['y'])
        elapsed_time = time.time() - start_time
        print('Logging Info - Training time: %s' % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        train_log['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    # load the best model
    model.load_best_model()

    if eval_on_train:
        # might take a long time
        print('Logging Info - Evaluate over train data:')
        train_acc = model.evaluate(x=train_input['x'], y=train_input['y'])
        train_log['train_acc'] = train_acc

    print('Logging Info - Evaluate over valid data:')
    valid_acc = model.evaluate(x=dev_input['x'], y=dev_input['y'])
    train_log['valid_acc'] = valid_acc

    print('Logging Info - Evaluate over test data...')
    test_acc = model.evaluate(x=test_input['x'], y=test_input['y'])
    train_log['test_acc'] = test_acc

    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG, genre), log=train_log, mode='a')
    return train_log


if __name__ == '__main__':
    model_names = ['KerasInfersent', 'KerasEsim', 'KerasSiameseBiLSTM', 'KerasSiameseCNN', 'KerasIACNN',
                   'KerasDecomposable']
    genres = ['mednli']
    input_levels = ['word']
    word_embed_types = ['glove_cc']
    word_embed_trainables = [False]
    batch_sizes = [32]
    learning_rates = [0.001]
    optimizer_types = ['adam']
    input_configs = ['elmo_id']
    elmo_output_modes = ['elmo']

    for model_name, genre, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate, optimizer, \
        input_config, elmo_output_mode in product(model_names, genres, input_levels, word_embed_types,
                                                  word_embed_trainables, batch_sizes, learning_rates, optimizer_types,
                                                  input_configs, elmo_output_modes):
        if model_name == 'KerasInfersent':
            for encoder_type in ['lstm', 'gru', 'bilstm', 'bigru', 'bilstm_max_pool', 'bilstm_mean_pool',
                                 'self_attentive', 'h_cnn']:
                train_model(genre, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate,
                            optimizer, model_name, overwrite=False, eval_on_train=False, input_config=input_config,
                            elmo_output_mode=elmo_output_mode, encoder_type=encoder_type)

        elif model_name == 'KerasDecomposable':
            for add in [True, False]:
                train_model(genre, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate,
                            optimizer, model_name, overwrite=False, eval_on_train=False, input_config=input_config,
                            elmo_output_mode=elmo_output_mode, add_intra_sentence_attention=add)

        train_model(genre, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate, optimizer,
                    model_name, overwrite=False, eval_on_train=False, input_config=input_config,
                    elmo_output_mode=elmo_output_mode)