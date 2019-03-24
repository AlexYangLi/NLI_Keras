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
from models.tfhub_bert_model import TFHubBertModel
from config import ModelConfig, PERFORMANCE_LOG, LOG_DIR, PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, \
    VOCABULARY_TEMPLATE, EXTERNAL_WORD_VECTORS_FILENAME, LABELS
from utils.data_loader import load_input_data
from utils.io import write_log, format_filename, pickle_load
from utils.cache import ELMoCache
from utils.data_generator import ELMoGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
                                                    '_'.join([str(k) + '_' + str(v) for k, v in kwargs.items()]))

    input_config = kwargs['input_config'] if 'input_config' in kwargs else 'token'  # input default is word embedding
    if input_config in ['cache_elmo', 'token_combine_cache_elmo']:
        # get elmo embedding based on cache, we first get a ELMoCache instance
        if 'elmo_model_type' in kwargs:
            elmo_model_type = kwargs['elmo_model_type']
            kwargs.pop('elmo_model_type')   # we don't need it in kwargs any more
        else:
            elmo_model_type = 'allennlp'
        if 'elmo_output_mode' in kwargs:
            elmo_output_mode = kwargs['elmo_output_mode']
            kwargs.pop('elmo_output_mode')
        else:
            elmo_output_mode ='elmo'
        elmo_cache = ELMoCache(options_file=config.elmo_options_file, weight_file=config.elmo_weight_file,
                               cache_dir=config.cache_dir, idx2token=config.idx2token,
                               max_sentence_length=config.max_len, elmo_model_type=elmo_model_type,
                               elmo_output_mode=elmo_output_mode)
    elif input_config in ['elmo_id', 'elmo_s', 'token_combine_elmo_id', 'token_combine_elmo_s']:
        # get elmo embedding using tensorflow_hub, we must provide a tfhub_url
        kwargs['elmo_model_url'] = config.elmo_model_url

    # logger to log output of training process
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
    model.summary()

    train_input, dev_input, test_input = None, None, None
    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    if not os.path.exists(model_save_path) or overwrite:
        start_time = time.time()

        if input_config in ['cache_elmo', 'token_combine_cache_elmo']:
            train_input = ELMoGenerator(genre, input_level, 'train', config.batch_size, elmo_cache,
                                        return_data=(input_config == 'token_combine_cache_elmo'))
            dev_input = ELMoGenerator(genre, input_level, 'dev', config.batch_size, elmo_cache,
                                      return_data=(input_config == 'token_combine_cache_elmo'))
            model.train_with_generator(train_input, dev_input)
        else:
            train_input = load_input_data(genre, input_level, 'train', input_config)
            dev_input = load_input_data(genre, input_level, 'dev', input_config)
            model.train(x_train=train_input['x'], y_train=train_input['y'], x_valid=dev_input['x'],
                        y_valid=dev_input['y'])
        elapsed_time = time.time() - start_time
        print('Logging Info - Training time: %s' % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        train_log['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    # load the best model
    model.load_best_model()

    if eval_on_train:
        # might take a long time
        print('Logging Info - Evaluate over train data:')
        if input_config in ['cache_elmo', 'token_combine_cache_elmo']:
            train_input = ELMoGenerator(genre, input_level, 'train', config.batch_size, elmo_cache,
                                        return_data=(input_config == 'token_combine_cache_elmo'),
                                        return_label=False)
            train_acc = model.evaluate_with_generator(generator=train_input, y=train_input.input_label)
        else:
            train_input = load_input_data(genre, input_level, 'train', input_config)
            train_acc = model.evaluate(x=train_input['x'], y=train_input['y'])
        train_log['train_acc'] = train_acc

    print('Logging Info - Evaluate over valid data:')
    if input_config in ['cache_elmo', 'token_combine_cache_elmo']:
        dev_input = ELMoGenerator(genre, input_level, 'dev', config.batch_size, elmo_cache,
                                  return_data=(input_config == 'token_combine_cache_elmo'),
                                  return_label=False)
        valid_acc = model.evaluate_with_generator(generator=dev_input, y=dev_input.input_label)
    else:
        if dev_input is None:
            dev_input = load_input_data(genre, input_level, 'dev', input_config)
        valid_acc = model.evaluate(x=dev_input['x'], y=dev_input['y'])
    train_log['valid_acc'] = valid_acc

    print('Logging Info - Evaluate over test data:')
    if input_config in ['cache_elmo', 'token_combine_cache_elmo']:
        test_input = ELMoGenerator(genre, input_level, 'test', config.batch_size, elmo_cache,
                                   return_data=(input_config == 'token_combine_cache_elmo'),
                                   return_label=False)
        test_acc = model.evaluate_with_generator(generator=test_input, y=test_input.input_label)
    else:
        if test_input is None:
            test_input = load_input_data(genre, input_level, 'test', input_config)
        test_acc = model.evaluate(x=test_input['x'], y=test_input['y'])
    train_log['test_acc'] = test_acc

    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG, genre), log=train_log, mode='a')
    return train_log


def train_bert(genre, input_level, batch_size):
    config = ModelConfig()
    config.genre = genre
    config.input_level = input_level
    config.max_len = config.word_max_len[genre] if input_level == 'word' else config.char_max_len[genre]
    config.batch_size = batch_size

    model = TFHubBertModel(config, [0, 1, 2], EXTERNAL_WORD_VECTORS_FILENAME['tfhub_bert'])

    train_input = load_input_data(genre, input_level, 'train', 'bert')
    valid_input = load_input_data(genre, input_level, 'valid', 'bert')
    test_input = load_input_data(genre, input_level, 'test', 'bert')
    model.train(train_input, valid_input)
    model.evaluate(valid_input)
    model.evaluate(test_input)


if __name__ == '__main__':
    # train_bert('mednli', 'word', 32)
    model_names = ['KerasInfersent', 'KerasEsim', 'KerasSiameseBiLSTM', 'KerasSiameseCNN', 'KerasIACNN',
                   'KerasDecomposable']
    genres = ['mednli']
    input_levels = ['word']
    word_embed_types = ['glove_cc']
    word_embed_trainables = [False]
    batch_sizes = [32]
    learning_rates = [0.001]
    optimizer_types = ['adam']
    input_configs = ['token_combine_elmo_id']
    elmo_output_modes = ['lstm_outputs1']

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
        else:
            train_model(genre, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate, optimizer,
                        model_name, overwrite=False, eval_on_train=True, input_config=input_config,
                        elmo_output_mode=elmo_output_mode)

    # model_names = ['KerasInfersent', 'KerasEsim', 'KerasSiameseBiLSTM', 'KerasSiameseCNN', 'KerasIACNN',
    #                'KerasDecomposable']
    # genres = ['mednli']
    # input_levels = ['word']
    # word_embed_types = ['glove_cc']
    # word_embed_trainables = [False]
    # batch_sizes = [64]
    # learning_rates = [0.001]
    # optimizer_types = ['adam']
    #
    # for model_name, genre, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate, optimizer \
    #         in product(model_names, genres, input_levels, word_embed_types,word_embed_trainables, batch_sizes,
    #                    learning_rates, optimizer_types):
    #     if model_name == 'KerasInfersent':
    #         for encoder_type in ['lstm', 'gru', 'bilstm', 'bigru', 'bilstm_max_pool', 'bilstm_mean_pool',
    #                              'self_attentive', 'h_cnn']:
    #             train_model(genre, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate,
    #                         optimizer, model_name, overwrite=False, eval_on_train=False, encoder_type=encoder_type)
    #     elif model_name == 'KerasDecomposable':
    #         for add in [True, False]:
    #             train_model(genre, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate,
    #                         optimizer, model_name, overwrite=False, eval_on_train=False,
    #                         add_intra_sentence_attention=add)
    #     train_model(genre, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate, optimizer,
    #                 model_name, overwrite=False, eval_on_train=False)

    # train_model('mednli', 'word', 'glove_cc', False, 128, 0.001, 'adam', 'KerasInfersent', overwrite=False,
    #             eval_on_train=False, encoder_type='self_attentive')

    # model_names = ['KerasInfersent', 'KerasEsim', 'KerasSiameseBiLSTM', 'KerasSiameseCNN', 'KerasIACNN',
    #                'KerasDecomposable']
    # genres = ['mednli']
    # input_levels = ['word']
    # word_embed_types = ['glove_cc']
    # word_embed_trainables = [False]
    # batch_sizes = [32]
    # learning_rates = [0.001]
    # optimizer_types = ['adam']
    # input_configs = ['cache_elmo']
    # elmo_output_modes = ['elmo_avg']
    # elmo_model_types = ['bilmtf']
    #
    # for model_name, genre, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate, optimizer, \
    #     input_config, elmo_output_mode, elmo_model_type in product(model_names, genres, input_levels, word_embed_types,
    #                                                                 word_embed_trainables, batch_sizes, learning_rates,
    #                                                                optimizer_types, input_configs, elmo_output_modes,
    #                                                                elmo_model_types):
    #     if model_name == 'KerasInfersent':
    #         for encoder_type in ['lstm', 'gru', 'bilstm', 'bigru', 'bilstm_max_pool', 'bilstm_mean_pool',
    #                              'self_attentive', 'h_cnn']:
    #             train_model(genre, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate,
    #                         optimizer, model_name, overwrite=False, eval_on_train=False, input_config=input_config,
    #                         elmo_model_type=elmo_model_type, elmo_output_mode=elmo_output_mode,
    #                         encoder_type=encoder_type)
    #
    #     elif model_name == 'KerasDecomposable':
    #         for add in [True, False]:
    #             train_model(genre, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate,
    #                         optimizer, model_name, overwrite=False, eval_on_train=False, input_config=input_config,
    #                         elmo_model_type=elmo_model_type, elmo_output_mode=elmo_output_mode,
    #                         add_intra_sentence_attention=add)
    #     else:
    #         train_model(genre, input_level, word_embed_type, word_embed_trainable, batch_size, learning_rate, optimizer,
    #                     model_name, overwrite=False, eval_on_train=False, input_config=input_config,
    #                     elmo_model_type=elmo_model_type, elmo_output_mode=elmo_output_mode)


