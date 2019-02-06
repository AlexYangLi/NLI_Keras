# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: embedding.py

@time: 2019/2/2 21:42

@desc:

"""

import logging
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


def load_glove_format(filename):
    word_vectors = {}
    embeddings_dim = -1
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split()

            try:
                word = line[0]
                word_vector = np.array([float(v) for v in line[1:]])
            except ValueError:
                continue

            if embeddings_dim == -1:
                embeddings_dim = len(word_vector)

            if len(word_vector) != embeddings_dim:
                continue

            word_vectors[word] = word_vector

    assert all(len(vw) == embeddings_dim for vw in word_vectors.values())

    return word_vectors, embeddings_dim


def load_trained(load_filename, vocabulary):
    if isinstance(load_filename, Path):
        load_filename = str(load_filename)  # gensim dosen't support opening Pathlib
    word_vectors = {}
    try:
        model = KeyedVectors.load_word2vec_format(load_filename)
        weights = model.wv.syn0
        embedding_dim = weights.shape[1]
        for k, v in model.wv.vocab.items():
            word_vectors[k] = weights[v.index, :]
    except ValueError:
        word_vectors, embedding_dim = load_glove_format(load_filename)

    emb = np.zeros(shape=(len(vocabulary) + 1, embedding_dim), dtype='float32')

    nb_unk = 0
    for w, i in vocabulary.items():
        if w not in word_vectors:
            nb_unk += 1
            emb[i, :] = np.random.normal(0, 0.05, embedding_dim)
        else:
            emb[i, :] = word_vectors[w]
    logging.info('From %s Embedding matrix created : %s, unknown tokens: %s', load_filename, emb.shape, nb_unk)
    return emb


def train_w2v(corpus, cut_func, vocabulary, embedding_dim=300):
    corpus = [cut_func(sentence) for sentence in corpus]
    model = Word2Vec(corpus, size=embedding_dim, min_count=1, window=5, sg=1, iter=10)
    weights = model.wv.syn0
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(vocabulary) + 1, embedding_dim), dtype='float32')

    nb_unk = 0
    for w, i in vocabulary.items():
        if w not in d:
            nb_unk += 1
            emb[i, :] = np.random.normal(0, 0.05, embedding_dim)
        else:
            emb[i, :] = weights[d[w], :]
    logging.info('Word2Vec Embedding matrix created: %s, unknown tokens: %s', emb.shape, nb_unk)
    return emb


def train_glove(corpus, vocabulary):
    pass


def train_fasttext(corpus, vocabulary):
    pass


def train_elmo(corpus, vocabulary):
    pass


def train_bert(corpus, vocabulary):
    pass


