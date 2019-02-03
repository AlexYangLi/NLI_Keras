# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: data_loader.py

@time: 2019/2/2 15:07

@desc:

"""
import json
import pandas as pd

from config import TRAIN_IDS_MATRIX_TEMPLATE, DEV_IDS_MATRIX_TEMPLATE, TEST_IDS_MATRIX_TEMPLATE
from utils.io import pickle_load


def read_nli_data(filename, set_genre=None):
    """
    Read NLI (unprocessed) data and return a DataFrame

    Optionally, set the genre column to `set_genre`
    """
    all_rows = []
    with open(str(filename)) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            if row['gold_label'] != '-':
                all_rows.append(row)
    nli_data = pd.DataFrame(all_rows)

    if set_genre is not None:
        nli_data['genre'] = set_genre

    return nli_data


def load_processed_data(genre, level, data_type):
    if data_type == 'train':
        filename = TRAIN_IDS_MATRIX_TEMPLATE.format(genre, level)
    elif data_type == 'valid' or data_type == 'dev':
        filename = DEV_IDS_MATRIX_TEMPLATE.format(genre, level)
    elif data_type == 'test':
        filename = TEST_IDS_MATRIX_TEMPLATE.format(genre, level)
    else:
        raise ValueError('Data Type Not Understood: {}'.format(data_type))
    return pickle_load(filename)
