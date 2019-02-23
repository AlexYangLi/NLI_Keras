# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: io.py

@time: 2019/2/1 13:56

@desc:

"""

import json
import numpy as np
import pickle


def format_filename(_dir, filename_template, *args):
    """Obtain the filename of data base on the provided template and parameters"""
    filename = _dir / filename_template.format(*args)
    return filename


def pickle_load(filename):
    try:
        with open(str(filename), 'rb') as f:
            obj = pickle.load(f)

        print('Logging Info - Loaded:', filename)

    except EOFError:
        print('Logging Error - Cannot load:', filename)
        obj = None

    return obj


def pickle_dump(filename, obj):
    with open(str(filename), 'wb') as f:
        pickle.dump(obj, f)

    print('Logging Info - Saved:', filename)


def write_log(filename, log, mode='w'):
    def default(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError

    with open(filename, mode) as writer:
        writer.write('\n')
        json.dump(log, writer, indent=4, default=default, ensure_ascii=False)



