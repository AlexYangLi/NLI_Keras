# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: config.py

@time: 2019/2/2 14:05

@desc:

"""

from keras.optimizers import Adam
from pathlib import Path

RAW_DATA_DIR = Path('raw_data')
PROCESSED_DATA_DIR = Path('./data')
LOG_DIR = Path('./log')
MODEL_SAVED_DIR = Path('./ckpt')
if not MODEL_SAVED_DIR.exists():
    MODEL_SAVED_DIR.mkdir()


SNLI_DIR = RAW_DATA_DIR / 'snli_1.0/'
SNLI_TRAIN_FILENAME = SNLI_DIR / 'snli_1.0_train.jsonl'
SNLI_DEV_FILENAME = SNLI_DIR / 'snli_1.0_dev.jsonl'
SNLI_TEST_FILENAME = SNLI_DIR / 'snli_1.0_test.jsonl'

MULTINLI_DIR = RAW_DATA_DIR / 'multinli_1.0/'
MULTINLI_TRAIN_FILENAME = MULTINLI_DIR / 'multinli_1.0_train.jsonl'
MULTINLI_DEV_FILENAME = MULTINLI_DIR / 'multinli_1.0_dev_matched.jsonl'

MLI_DIR = RAW_DATA_DIR / 'mednli_1.0/'
MLI_TRAIN_FILENAME = MLI_DIR / 'mli_train_v1.jsonl'
MLI_DEV_FILENAME = MLI_DIR / 'mli_dev_v1.jsonl'
MLI_TEST_FILENAME = MLI_DIR / 'mli_test_v1.jsonl'

TRAIN_DATA_TEMPLATE = 'genre_{}_train.pkl'
DEV_DATA_TEMPLATE = 'genre_{}_dev.pkl'
TEST_DATA_TEMPLATE = 'genre_{}_test.pkl'

TRAIN_IDS_MATRIX_TEMPLATE = 'genre_{}_level_{}_ids_train.pkl'
DEV_IDS_MATRIX_TEMPLATE = 'genre_{}_level_{}_ids_dev.pkl'
TEST_IDS_MATRIX_TEMPLATE = 'genre_{}_level_{}_ids_test.pkl'

EMBEDDING_MATRIX_TEMPLATE = 'genre_{}_type_{}_embeddings.npy'
TOKENIZER_TEMPLATE = 'genre_{}_level_{}_tokenizer.pkl'
VOCABULARY_TEMPLATE = 'genre_{}_level_{}_vocab.pkl'

ANALYSIS_LOG_TEMPLATE = 'genre_{}_analysis.log'
PERFORMANCE_LOG = 'genre_{}_performance.log'

EXTERNAL_WORD_VECTORS_DIR = RAW_DATA_DIR / 'word_embeddings/'
EXTERNAL_WORD_VECTORS_FILENAME = {
    'glove_cc': EXTERNAL_WORD_VECTORS_DIR / 'glove.840B.300d.txt',
    'fasttext_cc': EXTERNAL_WORD_VECTORS_DIR / 'fasttext-wiki-news-300d-1M-subword.vec',
    'fasttext_wiki': EXTERNAL_WORD_VECTORS_DIR / 'fasttext-crawl-300d-2M-subword.vec'
}

LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
GENRES = ['fiction', 'government', 'slate', 'telephone', 'travel', 'snli', 'multinli', 'mednli']


class ProcessConfig(object):
    def __init__(self):
        self.clean = False
        self.stem = False
        self.lowercase = True
        self.word_max_len = None
        self.char_max_len = None
        self.padding = 'post'
        self.truncating = 'post'
        self.n_class = 3
        self.word_cut_func = lambda x: x.split()
        self.char_cut_func = lambda x: list(x)


class ModelConfig(object):
    def __init__(self):
        # input configuration
        self.genre = 'snli'
        self.input_level = 'word'
        self.word_max_len = {'snli': 82, 'mednli': 202}
        self.char_max_len = {'snli': 406, 'mednli': 1132}
        self.max_len = 0
        self.word_embed_type = 'glove'
        self.word_embed_dim = 300
        self.word_embed_trainable = False
        self.word_embeddings = None

        # model structure configuration
        self.exp_name = None
        self.model_name = None
        self.rnn_units = 300
        self.dense_units = 128

        # model training configuration
        self.batch_size = 512
        self.n_epoch = 50
        self.learning_rate = 0.001
        self.optimizer = Adam(self.learning_rate)
        self.dropout = 0.5
        self.l2_reg = 0.001

        # output configuration
        self.n_class = 3

        # checkpoint configuration
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_acc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stoping configuration
        self.early_stopping_monitor = 'val_acc'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1
