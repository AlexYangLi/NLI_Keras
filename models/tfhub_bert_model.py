# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: tfhub_bert_model.py

@time: 2019/3/18 19:49

@desc:

"""

import bert
from bert import tokenization, run_classifier, optimization
import tensorflow_hub as hub
import tensorflow as tf


class TFHubBertModel(object):
    def __init__(self, config, label_list, hub_url=None):
        self.config = config
        self.n_class = config.n_class
        self.label_list = label_list
        self.max_len = config.max_len
        self.batch_size = config.batch_size
        self.output_dir = config.checkpoint_dir
        self.num_epoch = 3.0
        self.learning_rate = 2e-5
        self.warmup_proportion = 0.1
        self.save_checkpoints_steps = 500
        self.save_summary_steps = 100

        if hub_url is None:
            self.hub_url = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
        else:
            self.hub_url = hub_url

        self.tokenizer = self.create_tokenizer_from_hub_module()

        self.graph = tf.Graph()
        self.estimator = None

    def create_tokenizer_from_hub_module(self):
        """Get the vocab file and casing info from the Hub module."""
        with tf.Graph().as_default():
            bert_module = hub.Module(self.hub_url)
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                      tokenization_info["do_lower_case"]])
        return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    def create_model(self, is_predicting, input_ids, input_mask, segment_ids, labels):
        """Create a classification model"""
        bert_module = hub.Module(self.hub_url, trainable=True)
        bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        bert_outputs = bert_module(inputs=bert_inputs, signature='tokens', as_dict=True)

        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_outputs" for token-level output.
        output_layer = bert_outputs["pooled_output"]

        hidden_size = output_layer.shape[-1].value
        # Create our own layer to tune for politeness data.
        output_weights = tf.get_variable("output_weights", [self.n_class, hidden_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable("output_bias", [self.n_class], initializer=tf.zeros_initializer())

        with tf.variable_scope('loss'):
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(labels, depth=self.n_class, dtype=tf.float32)

            predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
            # If we're predicting, we want predicted labels and the probabiltiies.
            if is_predicting:
                return predicted_labels, log_probs

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return (loss, predicted_labels, log_probs)

    # model_fn_builder actually creates our model function
    # using the passed parameters for num_labels, learning_rate, etc.
    def model_fn_builder(self, learning_rate, num_train_steps, num_warmup_steps):

        def model_fn(features, labels, mode, params):
            """The `model_fn` for TPUEstimator."""

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]

            is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

            # TRAIN and EVAL
            if not is_predicting:
                loss, predicted_labels, log_probs = self.create_model(
                    is_predicting, input_ids, input_mask, segment_ids, label_ids)

                train_op = optimization.create_optimizer(
                    loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

                # Calculate evaluation metrics.
                def metric_fn(label_ids, predicted_labels):
                    accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                    return {
                        "eval_accuracy": accuracy,
                    }

                eval_metrics = metric_fn(label_ids, predicted_labels)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    return tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=loss,
                                                      train_op=train_op)
                else:
                    return tf.estimator.EstimatorSpec(mode=mode,
                                                      loss=loss,
                                                      eval_metric_ops=eval_metrics)
            else:
                predicted_labels, log_probs = self.create_model(is_predicting, input_ids, input_mask, segment_ids,
                                                                label_ids, num_labels)

                predictions = {
                    'probabilities': log_probs,
                    'labels': predicted_labels
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Return the actual model function in the closure
        return model_fn

    def build(self, num_train_steps, num_warmup_steps):
        run_config = tf.estimator.RunConfig(model_dir=self.output_dir, save_summary_steps=self.save_summary_steps,
                                            save_checkpoints_steps=self.save_checkpoints_steps)
        model_fn = self.model_fn_builder(learning_rate=self.learning_rate, num_train_steps=num_train_steps,
                                         num_warmup_steps=num_warmup_steps)
        self.estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config,
                                                params={"batch_size": self.batch_size})

    def train(self, train_examples, dev_examples):
        train_examples += dev_examples
        train_features = run_classifier.convert_examples_to_features(train_examples, self.label_list,
                                                                     self.max_len, self.tokenizer)

        num_train_steps = int(len(train_features) / self.batch_size * self.num_epoch)
        num_warmup_steps = int(num_train_steps * self.warmup_proportion)

        if self.estimator is None:
            self.build(num_train_steps, num_warmup_steps)

        train_input_fn = bert.run_classifier.input_fn_builder(features=train_features, seq_length=self.max_len,
                                                              is_training=True)
        print('Beginning Training!')
        self.estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print("Training end")

    def evaluate(self, examples):
        features = run_classifier.convert_examples_to_features(examples, self.label_list, self.max_len, self.tokenizer)
        input_fn = run_classifier.input_fn_builder(features=features, seq_length=self.max_len, is_training=False)
        self.estimator.evaluate(input_fn=input_fn, steps=None)

    def predict(self, examples):
        features = run_classifier.convert_examples_to_features(examples, self.label_list, self.max_len, self.tokenizer)
        input_fn = run_classifier.input_fn_builder(features=features, seq_length=self.max_len, is_training=False)
        predictions = self.estimator.predict(input_fn=input_fn)
        return predictions['probabilities']




