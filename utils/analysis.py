# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: analysis.py

@time: 2019/2/3 10:06

@desc:

"""


import logging
from collections import Counter
import numpy as np


def analyze_len_distribution(text_len):
    max_len = np.max(text_len)
    min_len = np.min(text_len)
    avg_len = np.average(text_len)
    median_len = np.median(text_len)
    logging.info('max len:', max_len, 'min_len', min_len, 'avg len', avg_len, 'median len', median_len)
    for i in range(int(median_len), int(max_len), 5):
        less = list(filter(lambda x: x <= i, text_len))
        ratio = len(less) / len(text_len)
        print(i, ratio)
        if ratio >= 0.99:
            break


def analyze_class_distribution(labels):
    for cls, count in Counter(labels).most_common():
        logging.info(cls, count, count / len(labels))

