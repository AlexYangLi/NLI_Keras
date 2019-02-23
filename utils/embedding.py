# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: embedding.py

@time: 2019/2/2 21:42

@desc:

"""

import os
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from fastText import train_unsupervised
from glove import Glove, Corpus


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
    print('Logging Info - From {} Embedding matrix created : {}, unknown tokens: {}'.format(load_filename, emb.shape,
                                                                                            nb_unk))
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
    print('Logging Info - Word2Vec Embedding matrix created: {}, unknown tokens: {}'.format(emb.shape, nb_unk))
    return emb


# here we use a python implementation of Glove, but the official glove implementation of C version is also highly
# recommended: https://github.com/stanfordnlp/GloVe/blob/master/demo.sh
def train_glove(corpus, cut_func, vocabulary, embedding_dim=300):
    corpus = [cut_func(sentence) for sentence in corpus]
    corpus_model = Corpus()
    corpus_model.fit(corpus, window=10)
    glove = Glove(no_components=embedding_dim, learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=10, no_threads=4, verbose=True)
    glove.add_dictionary(corpus_model.dictionary)

    emb = np.zeros(shape=(len(vocabulary) + 1, embedding_dim), dtype='float32')

    nb_unk = 0
    for w, i in vocabulary.items():
        if w not in glove.dictionary:
            nb_unk += 1
            emb[i, :] = np.random.normal(0, 0.05, embedding_dim)
        else:
            emb[i, :] = glove.word_vectors[glove.dictionary[w]]
    print('Logging Info - Glove Embedding matrix created: {}, unknown tokens: {}'.format(emb.shape, nb_unk))
    return emb


def train_fasttext(corpus, cut_func, vocabulary, embedding_dim=300):
    corpus = [' '.join(cut_func(sentence)) for sentence in corpus]
    corpus_file_path = 'fasttext_tmp_corpus.txt'
    with open(corpus_file_path, 'w', encoding='utf8')as writer:
        for sentence in corpus:
            writer.write(sentence + '\n')

    model = train_unsupervised(input=corpus_file_path, model='skipgram', epoch=10, minCount=1, wordNgrams=3, dim=300)

    model_vocab = model.get_words()

    emb = np.zeros(shape=(len(vocabulary) + 1, embedding_dim), dtype='float32')
    nb_unk = 0
    for w, i in vocabulary.items():
        if w not in model_vocab:
            nb_unk += 1
            emb[i, :] = np.random.normal(0, 0.05, embedding_dim)
        else:
            emb[i, :] = model.get_word_vector(w)
    print('Logging Info - Fasttext Embedding matrix created: {}, unknown tokens: {}'.format(emb.shape, nb_unk))
    os.remove(corpus_file_path)
    return emb


def train_elmo(corpus, vocabulary):
    pass


def train_bert(corpus, vocabulary):
    pass


