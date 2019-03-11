# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: preprocess.py

@time: 2019/2/1 14:05

@desc:

"""

import os
import itertools
import time
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from config import SNLI_TRAIN_FILENAME, SNLI_DEV_FILENAME, SNLI_TEST_FILENAME, MULTINLI_TRAIN_FILENAME, \
    MULTINLI_DEV_FILENAME, MLI_TRAIN_FILENAME, MLI_DEV_FILENAME, MLI_TEST_FILENAME, TRAIN_DATA_TEMPLATE, \
    DEV_DATA_TEMPLATE, TEST_DATA_TEMPLATE, TRAIN_IDS_MATRIX_TEMPLATE, DEV_IDS_MATRIX_TEMPLATE, \
    TEST_IDS_MATRIX_TEMPLATE, PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR, EXTERNAL_WORD_VECTORS_FILENAME, \
    EMBEDDING_MATRIX_TEMPLATE, TOKENIZER_TEMPLATE, VOCABULARY_TEMPLATE, ANALYSIS_LOG_TEMPLATE
from config import LABELS, GENRES
from config import ProcessConfig
from utils.data_loader import read_nli_data
from utils.text import get_tokens_from_parse, clean_data, stem_data
from utils.embedding import load_trained, train_w2v, train_glove, train_fasttext
from utils.analysis import analyze_len_distribution, analyze_class_distribution
from utils.io import pickle_dump, write_log, format_filename


def load_data():
    """Load SNLI, MultiNLI and MLI datasets into train/dev DataFrames"""
    data_snli_train, data_snli_dev, data_snli_test = None, None, None
    data_multinli_train, data_multinli_dev = None, None
    data_mli_train, data_mli_dev, data_mli_test = None, None, None

    # if SNLI_TRAIN_FILENAME.exists():
    #     data_snli_train = read_nli_data(SNLI_TRAIN_FILENAME, set_genre='snli')
    #     data_snli_dev = read_nli_data(SNLI_DEV_FILENAME, set_genre='snli')
    #     print('Logging Info - SNLI: train - {}, dev - {}'.format(data_snli_train.shape, data_snli_dev.shape))

    if MULTINLI_TRAIN_FILENAME.exists():
        data_multinli_train = read_nli_data(MULTINLI_TRAIN_FILENAME)
        data_multinli_dev = read_nli_data(MULTINLI_DEV_FILENAME)
        print('Logging Info - MultiNLI: train - {}, dev - {}'.format(data_multinli_train.shape, data_multinli_dev.shape))

    if MLI_TRAIN_FILENAME.exists():
        data_mli_train = read_nli_data(MLI_TRAIN_FILENAME, set_genre='mednli')
        data_mli_dev = read_nli_data(MLI_DEV_FILENAME, set_genre='mednli')
        print('Logging Info - MLI: train - {}, dev - {}'.format(data_mli_train.shape, data_mli_dev.shape))

    # if SNLI_TEST_FILENAME.exists():
    #     data_snli_test = read_nli_data(SNLI_TEST_FILENAME, set_genre='snli')
    #     print('Logging Info - SNLI: test - {}'.format(data_snli_test.shape))

    if MLI_TEST_FILENAME.exists():
        data_mli_test = read_nli_data(MLI_TEST_FILENAME, set_genre='mednli')
        print('Logging Info - MLI: test - {}'.format(data_mli_test.shape))

    # Drop columns that are presented not in all datasets
    columns_to_drop = ['captionID', 'promptID', 'annotator_labels']
    for d in [data_snli_train, data_snli_dev, data_snli_test, data_multinli_train, data_multinli_dev, data_mli_train,
              data_mli_dev, data_mli_test]:
        if d is not None:
            d.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

    # concatenate all data together
    data_train = pd.concat([data_snli_train, data_multinli_train, data_mli_train], axis=0)
    data_dev = pd.concat([data_snli_dev, data_multinli_dev, data_mli_dev], axis=0)
    data_test = pd.concat([data_snli_test, data_mli_test])

    data_train.set_index('genre', inplace=True)
    data_dev.set_index('genre', inplace=True)
    data_test.set_index('genre', inplace=True)

    return data_train, data_dev, data_test


def get_premise_hypothesis_label(data):
    labels = data['gold_label'].map(LABELS).tolist()
    premise_tokens = data['sentence1_binary_parse'].map(get_tokens_from_parse).tolist()
    hypothesis_tokens = data['sentence2_binary_parse'].map(get_tokens_from_parse).tolist()

    return {'premise': premise_tokens, 'hypothesis': hypothesis_tokens, 'label': labels}


def process_data(data, is_clean, is_stem):
    data = get_premise_hypothesis_label(data)
    print('Logging Info - Premise, hypothesis and label data got')

    if is_clean:
        data = clean_data(data)
        print('Logging Info - Data cleaned')
    if is_stem:
        data = stem_data(data)
        print('Logging Info - Data stemmed')

    return data


def create_token_ids_matrix(tokenizer, sequences, padding, truncating, max_len=None):
    tokens_ids = tokenizer.texts_to_sequences(sequences)

    # there might be zero len sequences - fix it by putting a random token there (or id 1 in the worst case)
    tokens_ids_flattened = list(itertools.chain.from_iterable(tokens_ids))
    max_id = max(tokens_ids_flattened) if len(tokens_ids_flattened) > 0 else -1

    text_lens = list()
    for i in range(len(tokens_ids)):
        if len(tokens_ids[i]) == 0:
            id_to_put = np.random.randint(1, max_id) if max_id != -1 else 1
            tokens_ids[i].append(id_to_put)
        text_lens.append(max(len(tokens_ids[i]), 1))

    print('Logging Info - pad sequence with max_len = %d' % max_len)
    tokens_ids = pad_sequences(tokens_ids, maxlen=max_len, padding=padding, truncating=truncating)
    return tokens_ids


def create_data_matrices(tokenizer, data, padding, truncating, n_class, max_len=None):
    premise = create_token_ids_matrix(tokenizer, data['premise'], padding, truncating, max_len)
    hypothesis = create_token_ids_matrix(tokenizer, data['hypothesis'], padding, truncating, max_len)
    label = to_categorical(data['label'], n_class)

    m_data = {
        'premise': premise,
        'hypothesis': hypothesis,
        'label': label,
    }
    return m_data


def main():
    process_conf = ProcessConfig()
    # create directory
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)

    # load SNLI, MultiNLI and MLI datasets
    data_train, data_dev, data_test = load_data()
    print('Logging Info - Data: train - {}, dev - {}, test - {}'.format(data_train.shape, data_dev.shape,
                                                                        data_test.shape))

    for genre in GENRES:
        if genre not in data_train.index:
            continue

        analyze_result = {}

        genre_train = data_train.loc[genre]
        genre_dev = data_dev.loc[genre]
        genre_test = data_test.loc[genre]   # might be None
        print('Logging Info - Genre: {}, train - {}, dev - {}, test - {}'.format(genre, genre_train.shape,
                                                                                 genre_dev.shape, genre_test.shape))
        analyze_result.update({'train_set': len(genre_train), 'dev_set': len(genre_dev),
                               'test_set': 0 if genre_test is None else len(genre_test)})

        genre_train_data = process_data(genre_train, process_conf.clean, process_conf.stem)
        genre_dev_data = process_data(genre_dev, process_conf.clean, process_conf.stem)

        # class distribution analysis
        train_label_distribution = analyze_class_distribution(genre_train_data['label'])
        analyze_result.update(dict(('train_cls_{}'.format(cls), percent) for cls, percent in train_label_distribution.items()))
        dev_label_distribution = analyze_class_distribution(genre_dev_data['label'])
        analyze_result.update(dict(('dev_cls_{}'.format(cls), percent) for cls, percent in dev_label_distribution.items()))

        # create tokenizer and vocabulary
        sentences_train = genre_train_data['premise'] + genre_train_data['hypothesis']
        sentences_dev = genre_dev_data['premise'] + genre_dev_data['hypothesis']

        word_tokenizer = Tokenizer(lower=process_conf.lowercase, filters='', char_level=False)
        char_tokenizer = Tokenizer(lower=process_conf.lowercase, filters='', char_level=True)
        word_tokenizer.fit_on_texts(sentences_train)    # just fit on train data
        char_tokenizer.fit_on_texts(sentences_train)
        print('Logging Info - Genre: {}, word_vocab: {}, char_vocab: {}'.format(genre, len(word_tokenizer.word_index),
                                                                                len(char_tokenizer.word_index)))
        analyze_result.update({'word_vocab': len(word_tokenizer.word_index),
                               'char_vocab': len(char_tokenizer.word_index)})

        # length analysis
        word_len_distribution, word_max_len = analyze_len_distribution(sentences_train, level='word')
        analyze_result.update(dict(('word_{}'.format(k), v) for k, v in word_len_distribution.items()))
        char_len_distribution, char_max_len = analyze_len_distribution(sentences_train, level='char')
        analyze_result.update(dict(('char_{}'.format(k), v) for k, v in char_len_distribution.items()))

        train_word_ids = create_data_matrices(word_tokenizer, genre_train_data, process_conf.padding,
                                              process_conf.truncating, process_conf.n_class, word_max_len)
        train_char_ids = create_data_matrices(char_tokenizer, genre_train_data, process_conf.padding,
                                              process_conf.truncating, process_conf.n_class, char_max_len)
        dev_word_ids = create_data_matrices(word_tokenizer, genre_dev_data, process_conf.padding,
                                            process_conf.truncating, process_conf.n_class, word_max_len)
        dev_char_ids = create_data_matrices(char_tokenizer, genre_dev_data, process_conf.padding,
                                            process_conf.truncating, process_conf.n_class, char_max_len)

        # create embedding matrix from pretrained word vectors
        glove_cc = load_trained(EXTERNAL_WORD_VECTORS_FILENAME['glove_cc'], word_tokenizer.word_index)
        fasttext_cc = load_trained(EXTERNAL_WORD_VECTORS_FILENAME['fasttext_cc'], word_tokenizer.word_index)
        fasttext_wiki = load_trained(EXTERNAL_WORD_VECTORS_FILENAME['fasttext_wiki'], word_tokenizer.word_index)
        # create embedding matrix by training on nil dataset
        w2v_nil = train_w2v(sentences_train+sentences_dev, lambda x: x.split(), word_tokenizer.word_index)
        c2v_nil = train_w2v(sentences_train+sentences_dev, lambda x: list(x), char_tokenizer.word_index)
        w_fasttext_nil = train_fasttext(sentences_train + sentences_dev, lambda x: x.split(), word_tokenizer.word_index)
        c_fasttext_nil = train_fasttext(sentences_train + sentences_dev, lambda x: list(x), char_tokenizer.word_index)
        w_glove_nil = train_glove(sentences_train + sentences_dev, lambda x: x.split(), word_tokenizer.word_index)
        c_glove_nil = train_glove(sentences_train + sentences_dev, lambda x: list(x), char_tokenizer.word_index)

        # save pre-process data
        pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, genre), genre_train_data)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_DATA_TEMPLATE, genre), genre_dev_data)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_IDS_MATRIX_TEMPLATE, genre, 'word'), train_word_ids)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, TRAIN_IDS_MATRIX_TEMPLATE, genre, 'char'), train_char_ids)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_IDS_MATRIX_TEMPLATE, genre, 'word'), dev_word_ids)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, DEV_IDS_MATRIX_TEMPLATE, genre, 'char'), dev_char_ids)

        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, genre, 'glove_cc'), glove_cc)
        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, genre, 'fasttext_cc'), fasttext_cc)
        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, genre, 'fasttext_wiki'), fasttext_wiki)
        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, genre, 'w2v_nil'), w2v_nil)
        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, genre, 'c2v_nil'), c2v_nil)
        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, genre, 'w_fasttext_nil'), w_fasttext_nil)
        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, genre, 'c_fasttext_nil'), c_fasttext_nil)
        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, genre, 'w_glove_nil'), w_glove_nil)
        np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, genre, 'c_glove_nil'), c_glove_nil)

        pickle_dump(format_filename(PROCESSED_DATA_DIR, TOKENIZER_TEMPLATE, genre, 'word'), word_tokenizer)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, TOKENIZER_TEMPLATE, genre, 'char'), char_tokenizer)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, genre, 'word'), word_tokenizer.word_index)
        pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, genre, 'char'), char_tokenizer.word_index)

        if genre_test is not None:
            genre_test_data = process_data(genre_test, process_conf.clean, process_conf.stem)
            test_label_distribution = analyze_class_distribution(genre_test_data['label'])
            analyze_result.update(
                dict(('test_cls_%d' % cls, percent) for cls, percent in test_label_distribution.items()))

            test_word_ids = create_data_matrices(word_tokenizer, genre_test_data, process_conf.padding,
                                                 process_conf.truncating, process_conf.n_class,
                                                 word_max_len)
            test_char_ids = create_data_matrices(char_tokenizer, genre_test_data, process_conf.padding,
                                                 process_conf.truncating, process_conf.n_class,
                                                 char_max_len)
            pickle_dump(format_filename(PROCESSED_DATA_DIR, TEST_DATA_TEMPLATE, genre), genre_test_data)
            pickle_dump(format_filename(PROCESSED_DATA_DIR, TEST_IDS_MATRIX_TEMPLATE, genre, 'word'), test_word_ids)
            pickle_dump(format_filename(PROCESSED_DATA_DIR, TEST_IDS_MATRIX_TEMPLATE, genre, 'char'), test_char_ids)

        # save analyze result
        analyze_result['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        write_log(format_filename(LOG_DIR, ANALYSIS_LOG_TEMPLATE, genre), analyze_result)


if __name__ == '__main__':
    main()
