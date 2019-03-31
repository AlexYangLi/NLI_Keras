# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: features.py

@time: 2019/2/1 14:02

@desc:

"""

import numpy as np
import textdistance
from fuzzywuzzy import fuzz
from simhash import Simhash


# length difference in word & char level
def length_distance(s1: str, s2: str, word_cut_func=None):
    s1_words = s1.split() if word_cut_func is None else word_cut_func(s1)
    s2_words = s2.split() if word_cut_func is None else word_cut_func(s2)
    return [len(s1_words), len(s2_words), len(s1_words)-len(s2_words), len(s1), len(s2), len(s1)-len(s2)]


# longest common sequence
def lcs_seq(s1: str, s2: str):
    result = [[0 for _ in range(len(s2)+1)] for _ in range(len(s1) + 1)]
    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            if s1[i-1] == s2[j-1]:  # be careful with index
                result[i][j] = result[i-1][j-1] + 1
            else:
                result[i][j] = max(result[i-1][j], result[i][j-1])
    return result[len(s1)][len(s2)]


def lcs_seq_norm(s1: str, s2: str):
    # return textdistance.lcsseq.normalized_similarity(s1, s2)
    l = lcs_seq(s1, s2)
    return 2 * l / (len(s1) + len(s2))


# longest common substring
def lcs_str(s1: str, s2: str):
    max_len = 0
    result = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            if s1[i-1] == s2[j-1]:  # be careful with index
                result[i][j] = result[i-1][j-1] + 1
            max_len = max(max_len, result[i][j])
    return max_len


def lcs_str_norm(s1, s2):
    l = lcs_str(s1, s2)
    return 2 * l / (len(s1) + len(s2))


# longest common substring with o(1) space complexity
def lcs_str_1(s1: str, s2: str):
    max_len = 0
    m, n = len(s1), len(s2)
    row, column = 0, n-1
    while row < m:
        i, j = row, column
        tmp_len = 0
        while i < m and j < n:
            if s1[i] == s2[j]:
                tmp_len += 1
            else:
                tmp_len = 0
            max_len = max(max_len, tmp_len)
            i += 1
            j += 1
        if column > 0:
            column -= 1
        else:
            row += 1
    return max_len


def lcs_str_1_norm(s1, s2):
    l = lcs_str_1(s1, s2)
    return 2 * l / (len(s1) + len(s2))


def edit_distance(s1: str, s2: str):
    result = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    for i in range(len(s1) + 1):
        result[i][0] = i
    for j in range(len(s2) + 1):
        result[0][j] = j

    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):
            if s1[i-1] == s2[j-1]:
                result[i][j] = result[i-1][j-1]
            else:
                result[i][j] = min(result[i-1][j], result[i][j-1], result[i-1][j-1]) + 1

    return result[len(s1)][len(s2)]


def jaro_distance(s1: str, s2: str):
    # if len(s1) > len(s2):
    #     max_str, min_str = s1, s2
    # else:
    #     max_str, min_str = s2, s1
    # match_window = max(math.floor(len(max_str) / 2) - 1, 0)
    #
    # # find matches indexes
    # match_indexes = [-1] * len(min_str)
    # match_flags = [False] * len(max_str)
    # matches = 0
    # for i in range(len(min_str)):
    #     c = min_str[i]
    #     start = max(i-match_window, 0)
    #     end = min(i+match_window+1, len(max_str))
    #     for j in range(start, end):
    #         if not match_flags[j] and c == max_str[j]:
    #             match_flags[j] = True
    #             match_indexes[i] = j
    #             matches += 1
    #             break
    #
    # if matches == 0:
    #     return 0
    #
    # # find matches, keep the order
    # ms1 = [min_str[i] for i in range(len(min_str)) if match_indexes[i] != -1]
    # ms2 = [max_str[i] for i in range(len(max_str)) if match_flags[i]]
    #
    # # get transpositions
    # transpositions = 0
    # for i in range(len(ms1)):
    #     if ms1[i] != ms2[i]:
    #         transpositions += 1
    #
    # final_result = (matches / len(s1) + matches / len(s2) + (matches - transpositions / 2) / matches) / 3
    return textdistance.jaro.similarity(s1, s2)


def jaro_winkler_dist(s1: str, s2: str, scaling=0.1):
    # jaro_dist = jaro_distance(s1, s2)
    # # find prefix
    # prefix = 0
    # for i in range(min(len(s1), len(s2))):
    #     if s1[i] == s2[i]:
    #         prefix += 1
    #     else:
    #         break
    # final_result = jaro_dist + prefix * scaling * (1-jaro_dist)
    return textdistance.jaro_winkler.similarity(s1, s2)


def other_distance(s1, s2):
    return [textdistance.hamming.normalized_similarity(s1, s2),
            textdistance.mlipns.normalized_similarity(s1, s2),
            textdistance.damerau_levenshtein.normalized_similarity(s1, s2),
            textdistance.strcmp95.normalized_similarity(s1, s2),
            textdistance.needleman_wunsch.normalized_similarity(s1, s2),
            textdistance.gotoh.normalized_similarity(s1, s2),
            textdistance.smith_waterman.normalized_similarity(s1, s2),
            textdistance.ratcliff_obershelp.normalized_similarity(s1, s2)]


def fuzzy(s1, s2):
    return [fuzz.ratio(s1, s2) / 100,
            fuzz.partial_ratio(s1, s2) / 100,
            fuzz.token_sort_ratio(s1, s2) / 100,
            fuzz.partial_token_sort_ratio(s1, s2) / 100,
            fuzz.token_set_ratio(s1, s2) / 100,
            fuzz.partial_token_set_ratio(s1, s2) / 100,
            fuzz.QRatio(s1, s2) / 100,
            fuzz.WRatio(s1, s2) / 100]


def simhash(s1, s2):
    return Simhash(s1).distance(Simhash(s2))


def char_ngram_overlap(s1, s2, ngram_range=range(1, 5)):
    def ngram(s, n):
        return set(s[i:i + n] for i in range(len(s) - n + 1))

    overlap_ratio = []
    for n in ngram_range:
        ngram_s1 = ngram(s1, n)
        n_gram_s2 = ngram(s2, n)
        overlap_ratio.append(2 * len(ngram_s1 & n_gram_s2) / (len(ngram_s1) + len(n_gram_s2)))
    return overlap_ratio


def word_ngram_overlap(s1, s2, ngram_range=(1, 3), word_cut_func=None):
    def ngram(s, n):
        return set(' '.join(s[i:i + n]) for i in range(len(s) - n + 1))

    s1 = s1.split() if word_cut_func is None else word_cut_func(s1)
    s2 = s2.split() if word_cut_func is None else word_cut_func(s2)
    overlap_ratio = []
    for n in ngram_range:
        ngram_s1 = ngram(s1, n)
        n_gram_s2 = ngram(s2, n)
        overlap_ratio.append(2 * len(ngram_s1 & n_gram_s2) / (len(ngram_s1) + len(n_gram_s2)))
    return overlap_ratio


def weighted_word_ngram_overlap(s1, s2, ngram_range=(1, 3)):
    """
    :param s1: [(w11, idf11, tf-idf11), (w12, idf12, tf-idf12), ..., ]
    :param s2: [(w21, idf21, tf-idf11), (w22, idf22, tf-idf22), ..., ]
    """
    words_1 = [word[0] for word in s1]
    words_2 = [word[0] for word in s2]
    idfs_1 = [word[1] for word in s1]
    idfs_2 = [word[1] for word in s2]
    tfidfs_1 = [word[2] for word in s1]
    tfidfs_2 = [word[2] for word in s2]

    def ngram(words, idfs, tfidfs, n):
        ngram_idfs, ngram_tfidfs = dict(), dict()
        for i in range(len(words) - n + 1):
            idf = np.mean(idfs[i:i+n])
            tfidf = np.mean(tfidfs[i:i+n])
            _ngram = ' '.join(words[i:i+n])
            ngram_idfs[_ngram] = idf
            ngram_tfidfs[_ngram] = tfidf
        return ngram_idfs, ngram_tfidfs

    overlap_ratio = []
    for n in ngram_range:
        ngram_idfs_1, ngram_tfidfs_1 = ngram(words_1, idfs_1, tfidfs_1, n)
        ngram_idfs_2, ngram_tfidfs_2 = ngram(words_2, idfs_2, tfidfs_2, n)
        overlap_ngram = set(ngram_idfs_1.keys()) & set(ngram_idfs_2.keys())
        overlap_idfs = [ngram_idfs_1[_ngram]+ngram_idfs_2[_ngram] for _ngram in overlap_ngram]
        overlap_tfidfs = [ngram_tfidfs_1[_ngram]+ngram_tfidfs_2[_ngram] for _ngram in overlap_ngram]
        overlap_ratio.append(sum(overlap_idfs) / (sum(ngram_idfs_1.values()) + sum(ngram_idfs_2.values())))
        overlap_ratio.append(sum(overlap_tfidfs) / (sum(ngram_tfidfs_1.values()) + sum(ngram_tfidfs_2.values())))
    return overlap_ratio


