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
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from fastText import train_unsupervised
from glove import Glove, Corpus

import tensorflow as tf
import tensorflow_hub as hub
from allennlp.commands.elmo import ElmoEmbedder


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


def load_elmo_from_tfhub(idx2token, token_ids, hub_url=None):
    """input sentence are processed token id sequences"""
    idx2token[0] = ''   # pad position, must add
    word_mapping = [x[1] for x in sorted(idx2token.items(), key=lambda x: x[0])]
    lookup_table = tf.contrib.lookup.index_to_string_table_from_tensor(word_mapping, default_value="<UNK>")
    if hub_url is None:
        hub_url = 'https://tfhub.dev/google/elmo/2'
    print('Logging Info - ')
    elmo = hub.Module(hub_url, trainable=False)

    inputs = tf.cast(token_ids, dtype=tf.int64)
    sequence_lengths = tf.cast(tf.count_nonzero(inputs, axis=1), dtype=tf.int32)
    embeddings = elmo(inputs={'tokens': lookup_table.lookup(inputs), 'sequence_len': sequence_lengths},
                      signature="tokens", as_dict=True)
    output_mask = tf.expand_dims(tf.cast(tf.not_equal(inputs, 0), tf.float32), axis=-1)
    embeddings *= output_mask

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        elmo_embeddings = sess.run(embeddings)

    return elmo_embeddings


def load_elmo_from_allennlp(idx2token, token_ids, options_file, weight_file, cuda_device=0):
    """input sentence are processed token id sequences"""
    print('Logging Info - Loading elmo from pre-trained model using ElmoEmbedder')
    elmo = ElmoEmbedder(options_file=options_file, weight_file=weight_file, cuda_device=cuda_device)

    input_tokens = []
    for sample in token_ids:
        token = []
        for idx in sample:
            if idx == 0 or idx not in idx2token:
                continue
            token.append(idx2token(idx))
        input_tokens.append(token)

    elmo_embeddings = elmo.embed_batch(input_tokens)

    return elmo_embeddings


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

