# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: data_loader.py

@time: 2019/2/2 15:07

@desc:

"""
import json
import numpy as np
import pandas as pd
from bert import run_classifier

from config import PROCESSED_DATA_DIR, TRAIN_IDS_MATRIX_TEMPLATE, DEV_IDS_MATRIX_TEMPLATE, TEST_IDS_MATRIX_TEMPLATE, \
    TRAIN_DATA_TEMPLATE, DEV_DATA_TEMPLATE, TEST_DATA_TEMPLATE
from utils.io import pickle_load, format_filename



def read_nli_data(filename, set_genre=None):
    """
    Read NLI (unprocessed) data and return a DataFrame

    Optionally, set the genre column to `set_genre`
    """
    all_rows = []
    with open(filename) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            if row['gold_label'] != '-':
                all_rows.append(row)
    nli_data = pd.DataFrame(all_rows)

    if set_genre is not None:
        nli_data['genre'] = set_genre

    return nli_data


def load_processed_text_data(genre, data_type):
    if data_type == 'train':
        filename = format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, genre)
    elif data_type == 'valid' or data_type == 'dev':
        filename = format_filename(PROCESSED_DATA_DIR, DEV_DATA_TEMPLATE, genre)
    elif data_type == 'test':
        filename = format_filename(PROCESSED_DATA_DIR, TEST_DATA_TEMPLATE, genre)
    else:
        raise ValueError('Data Type Not Understood: {}'.format(data_type))
    return pickle_load(filename)


def load_processed_data(genre, level, data_type):
    if data_type == 'train':
        filename = format_filename(PROCESSED_DATA_DIR, TRAIN_IDS_MATRIX_TEMPLATE, genre, level)
    elif data_type == 'valid' or data_type == 'dev':
        filename = format_filename(PROCESSED_DATA_DIR, DEV_IDS_MATRIX_TEMPLATE, genre, level)
    elif data_type == 'test':
        filename = format_filename(PROCESSED_DATA_DIR, TEST_IDS_MATRIX_TEMPLATE, genre, level)
    else:
        raise ValueError('Data Type Not Understood: {}'.format(data_type))
    return pickle_load(filename)


# load model input data
def load_input_data(genre, level, data_type, input_config):
    if input_config in ['token', 'elmo_id', 'token_combine_elmo_id']:
        _data = load_processed_data(genre, level, data_type)
        input_data = {'x': [_data['premise'], _data['hypothesis']], 'y': _data['label']}
    elif input_config == 'elmo_s':
        _data = load_processed_text_data(genre, data_type)
        _data['premise'] = np.array(_data['premise'], dtype=object)[:, np.newaxis]
        _data['hypothesis'] = np.array(_data['hypothesis'], dtype=object)[:, np.newaxis]
        _data['label'] = load_processed_data(genre, level, data_type)['label']
        input_data = {'x': [_data['premise'], _data['hypothesis']], 'y': _data['label']}
    elif input_config == 'token_combine_elmo_s':
        _data = load_processed_data(genre, level, data_type)
        _text_data = load_processed_text_data(genre, data_type)
        _text_data['premise'] = [np.array(_text_data['premise'], dtype=object)[:, np.newaxis]]
        _text_data['hypothesis'] = [np.array(_text_data['hypothesis'], dtype=object)[:, np.newaxis]]
        input_data = {'x': [_data['premise'], _data['hypothesis'], _text_data['premise'], _text_data['hypothesis']],
                      'y': _data['label']}
    elif input_config == 'bert':
        # prepare input examples for bert model
        _data = load_processed_text_data(genre, data_type)
        if _data['label'] is None:
            input_data = [run_classifier.InputExample(guid=None, text_a=p, text_b=h) for p, h
                          in zip(_data['premise'], _data['hypothesis'])]
        else:
            input_data = [run_classifier.InputExample(guid=None, text_a=p, text_b=h, label=l) for p, h, l
                          in zip(_data['premise'], _data['hypothesis'], _data['label'])]
    else:
        raise ValueError('input config Not Understood: {}'.format(input_config))
    return input_data



