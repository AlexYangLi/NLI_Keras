# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: io.py

@time: 2019/2/1 13:56

@desc:

"""

import logging
from collections import Iterable
import pickle


def format_processed_filename(_dir, filename_template, **kwargs):
    """Obtain the filename of the processed data base on the provided template and parameters"""
    filename = _dir / filename_template. format(**kwargs)
    return filename


def pickle_load(filename):
    try:
        with open(str(filename), 'rb') as f:
            obj = pickle.load(f)

        logging.info('Loaded: %s', filename)

    except EOFError:
        logging.warning('Cannot load: %s', filename)
        obj = None

    return obj


def pickle_dump(filename, obj):
    with open(str(filename), 'wb') as f:
        pickle.dump(obj, f)

    logging.info('Saved: %s', filename)


def save_log(filename, log):
    with open(filename, 'w') as writer:
        if isinstance(log, Iterable):
            for _log in log:
                writer.write(_log)
                writer.write('\n')
        else:
            writer.write(log)
            writer.write('\n')




