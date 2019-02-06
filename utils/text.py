# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: text.py

@time: 2019/2/1 14:01

@desc:

"""

import re
from nltk.stem.porter import PorterStemmer


def get_tokens_from_parse(parse):
    """Parse a string in the binary tree SNLI format and return a string of joined by space tokens"""
    cleaned = parse \
        .replace('(', ' ').replace(')', ' ') \
        .replace('-LRB-', '(').replace('-RRB-', ')') \
        .replace('-LSB-', '[').replace('-RSB-', ']')

    tokens = cleaned.split()

    cleaned_string = ' '.join(tokens)

    # remove all non-ASCII characters for MetaMap
    cleaned_string = cleaned_string.encode('ascii', errors='ignore').decode()

    return cleaned_string


def is_time(token):
    without_time = re.sub(r'(\d)*(\d):(\d\d)([aA][mM]|[pP][Mm])', '', token).strip()
    return not without_time


def is_num(token):
    try:
        num = float(token)
        return True
    except ValueError as e:
        return False


def is_non_alpha(token):
    without_alpha = re.sub(r'[^A-Za-z]', '', token).strip()
    return not without_alpha


def remove_punctuation(token):
    without_punctuation = re.sub(r'[^A-Za-z0-9]', '', token)
    return without_punctuation


def clean_sentence(sentence):
    tokens = sentence.split(' ')

    sentence_cleaned_tokens = []
    for i, token in enumerate(tokens):
        if is_time(token):
            sentence_cleaned_tokens.append('TIME')
            continue

        if is_num(token):
            sentence_cleaned_tokens.append('NUM')
            continue

        if is_non_alpha(token):
            continue

        token = remove_punctuation(token)
        sentence_cleaned_tokens.append(token)

    sentence_cleaned = ' '.join(sentence_cleaned_tokens)

    return sentence_cleaned


def clean_list_of_sentences(sentences):
    sentences_cleaned = [clean_sentence(s) for s in sentences]
    return sentences_cleaned


def clean_data(data):
    if data is None:
        return None

    premise_cleaned = clean_list_of_sentences(data['premise'])
    hypothesis_cleaned = clean_list_of_sentences(data['hypothesis'])

    data['premise'] = premise_cleaned
    data['hypothesis'] = hypothesis_cleaned

    return data


def stem_list_of_sentences(sentences, stemmer):
    sentences = [s.split() for s in sentences]
    sentences = [[stemmer.stem(t) for t in s] for s in sentences]
    sentences = [' '.join(s) for s in sentences]

    return sentences


def stem_data(data):
    if data is None:
        return None

    stemmer = PorterStemmer()

    premise_stemmed = stem_list_of_sentences(data['premise'], stemmer)
    hypothesis_stemmed = stem_list_of_sentences(data['hypothesis'], stemmer)

    data['premise'] = premise_stemmed
    data['hypothesis'] = hypothesis_stemmed

    return data


def lowecase_list_of_sentences(sentences):
    sentences_lowercased = [s.lower() for s in sentences]
    return sentences_lowercased


def lowercase_data(data):
    if data is None:
        return None

    premise_lowercased = lowecase_list_of_sentences(data['premise'])
    hypothesis_lowercased = lowecase_list_of_sentences(data['hypothesis'])

    data['premise'] = premise_lowercased
    data['hypothesis'] = hypothesis_lowercased

    return data
