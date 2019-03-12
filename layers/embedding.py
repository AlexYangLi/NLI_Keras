# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: embedding.py

@time: 2019/3/10 15:39

@desc:

"""


import tensorflow as tf
import tensorflow_hub as hub
from keras.engine.topology import Layer
from keras import backend as K


class ELMoEmbedding(Layer):
    """
    integrate ELMo Embeddings from tensorflow hub into a custom Keras layer, supporting weight update
    reference:  https://github.com/strongio/keras-elmo
                https://github.com/JHart96/keras_elmo_embedding_layer/blob/master/elmo.py
                https://tfhub.dev/google/elmo/2
    """
    def __init__(self, output_mode, idx2word=None, max_length=None, mask_zero=False, hub_url=None, elmo_trainable=None,
                 **kwargs):
        """
        inputs to ELMoEmbedding can be untokenzied sentences (shaped [batch_size, 1], typed string) or tokenzied word's
        id sequences (shaped [batch_size, max_length], typed int).
        When use untokenized sentences as input, max_length must be provided.
        When use word id sequences as input, idx2word must be provided to convert word id to word.
        """
        self.output_mode = output_mode
        if self.output_mode not in ['word_embed', 'lstm_output1', 'lstm_output2', 'elmo', 'default']:
            raise ValueError('Output Type Not Understood:`{}`'.format(self.output_mode))
        self.idx2word = idx2word
        self.max_length = max_length
        self.mask_zero = mask_zero
        self.dimension = 1024

        self.input_type = None
        self.word_mapping = None
        self.lookup_table = None

        # load elmo model locally by providing a local path due to the huge delay of downloading the model
        # for more information, see:
        # https://stackoverflow.com/questions/50322001/how-to-save-load-a-tensorflow-hub-module-to-from-a-custom-path
        # https://www.tensorflow.org/hub/hosting
        if hub_url is not None:
            self.hub_url = hub_url
        else:
            self.hub_url = 'https://tfhub.dev/google/elmo/2'
        if elmo_trainable is not None:
            self.trainable = elmo_trainable
        else:
            self.trainable = True if self.output_mode == 'elmo' else False
        self.elmo = None

        super(ELMoEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        if input_shape[1] == 1:
            self.input_type = 'sentence'
            assert self.max_length is not None
        else:
            self.input_type = 'word_id'
            self.max_length = input_shape[1]
            assert self.idx2word is not None
            self.idx2word[0] = ''   # padded position, must add
            self.word_mapping = [x[1] for x in sorted(self.idx2word.items(), key=lambda x: x[0])]
            self.lookup_table = tf.contrib.lookup.index_to_string_table_from_tensor(self.word_mapping,
                                                                                    default_value="<UNK>")
            self.lookup_table.init.run(session=K.get_session())

        print('Logging Info - Loading elmo from tensorflow hub....')
        self.elmo = hub.Module(self.hub_url, trainable=self.trainable, name="{}_elmo_hub".format(self.name))

        if self.trainable:
            self.trainable_weights += K.tf.trainable_variables(scope="^{}_elmo_hub/.*".format(self.name))

    def call(self, inputs, mask=None):
        if self.input_type == 'sentence':
            # inputs are untokenized sentences
            embeddings = self.elmo(inputs=K.squeeze(K.cast(inputs, tf.string), axis=1),
                                   signature="default", as_dict=True)[self.output_mode]
            elmo_max_length = K.int_shape(embeddings)[1]
            if self.max_length > elmo_max_length:
                embeddings = K.temporal_padding(embeddings, padding=(0, self.max_length-elmo_max_length))
            elif elmo_max_length > self.max_length:
                # embeddings = tf.slice(embeddings, begin=[0, 0, 0], size=[-1, self.max_length, -1])
                embeddings = embeddings[:, :self.max_length, :]     # more pythonic
        else:
            # inputs are tokenized word id sequence
            # convert inputs to word sequence
            inputs = tf.cast(inputs, dtype=tf.int64)
            sequence_lengths = tf.cast(tf.count_nonzero(inputs, axis=1), dtype=tf.int32)
            embeddings = self.elmo(inputs={'tokens': self.lookup_table.lookup(inputs),
                                           'sequence_len': sequence_lengths},
                                   signature="tokens", as_dict=True)[self.output_mode]
            if self.output_mode != 'defalut':
                output_mask = K.expand_dims(K.not_equal(inputs, 0), axis=-1)
                embeddings *= output_mask

        return embeddings

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero or self.input_type == 'sentence' or self.output_mode == 'default':
            # hard to compute mask when using sentences as input
            return None
        output_mask = K.not_equal(inputs, 0)
        return output_mask

    def compute_output_shape(self, input_shape):
        if self.output_mode == 'default':
            return input_shape[0], self.dimension
        else:
            return input_shape[0], self.max_length, self.dimension




