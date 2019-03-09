# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: analysis.py

@time: 2019/2/3 10:06

@desc:

"""

from collections import Counter
import numpy as np


def analyze_len_distribution(sentences, level):
    if level == 'word':
        text_len = [len(sentence.split()) for sentence in sentences]
    elif level == 'char':
        text_len = [len(list(sentence)) for sentence in sentences]
    else:
        raise ValueError('Level Not Understood: {}'.format(level))

    len_dist = dict()

    max_len = np.max(text_len)
    min_len = np.min(text_len)
    avg_len = np.average(text_len)
    median_len = np.median(text_len)
    print('Logging Info - max len: %d, min_len: %d, avg_len: %2f, median_len: %2f' % (max_len, min_len, avg_len,
                                                                                      median_len))
    len_dist.update({'max len:': max_len, 'min_len': min_len, 'avg len': avg_len, 'median len': median_len})

    _start_log_ratio = 0.95
    for i in range(int(median_len), int(max_len), 2):
        less = list(filter(lambda x: x <= i, text_len))
        ratio = len(less) / len(text_len)
        print('Logging Info - len: %d, ratio: %2f' % (i, ratio))

        if ratio >= _start_log_ratio:
            len_dist[i] = ratio
            _start_log_ratio += 0.01
        if ratio >= 0.99:
            break
    return len_dist, max_len


def analyze_class_distribution(labels):
    class_dist = dict()
    for cls, count in Counter(labels).most_common():
        print('Logging Info - class: {}, count: {}, ratio: {}'.format(cls, count, count / len(labels)))
        class_dist[cls] = count / len(labels)
    return class_dist

