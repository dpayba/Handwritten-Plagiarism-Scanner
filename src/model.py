import os
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from load_data import Batch

class Model:
    def __init__(self, char_list, must_restore, dump):
        self.dump = dump
        self.char_list = char_list
        self.must_restore = must_restore
        self.snap_ID = 0

        self.input_images = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None))

        self.create_cnn()
        self.create_rnn()

        self.batches_trained = 0
        self.update_operations = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_operations):
            self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)

        self.sess, self.saver = self.setup_tf()

    def create_cnn(self):
        cnn_4_dimension = tf.expand_dims(input=self.input_imgs, axis=3)

        # layer parameters
        kernel = [5, 5, 3, 3, 3]
        features = [1, 32, 64, 128, 128, 256]
        strides = pool = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        n_layers = len(strides)

        pool = cnn_4_dimension
        for i in range(n_layers):
            kernel = tf.Variable(tf.random.truncated_normal([kernel[i], kernel[i], features[i], features[i+1]],
                                                            stddev = 0.1))
            conv = tf.nn.conv2d(input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            conv_normalize = tf.compat.v1.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.nn.relu(conv_normalize)
            pool = tf.nn.max_pool2d(input=relu, ksize=(1, pool[i][0], pool[i][1], 1),
                                    strides=(1, strides[i][0], strides[i][1], 1), padding='VALID')

        self.cnn_output = pool

    def create_rnn(self):
        rnn_3_dimension = tf.squeeze(self.cnn_output, axis=[2])
        n_hidden = 256
        cells = [tf.compat.v1.rnn_cell.LSTMCell(num_units=n_hidden, state_is_tuple=True) for _ in
                 range(2)]  # 2 layers

        # stack cells basic
        stack_cells = tf.compat.v1.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional rnn
        (fw, bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=stack_cells, cell_bw=stack_cells, inputs=rnn_3_dimension,
                                                                dtype=rnn_3_dimension.dtype)
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        kernel = tf.Variable(tf.random.truncated_normal([1, 1, n_hidden*2, len(self.char_list) + 1], stddev=0.1))
        self.rnn_output = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'),
                                     axis=[2])

    def to_sparse(self, texts):
        # add ground truth to sparse for ctc loss
        indices = []
        values = []
        shape = [len(texts), 0]

        for element, text in enumerate(texts):
            label_s = [self.char_list.index(c) for c in text]
            if len(label_s) > shape[1]:
                shape[1] = len(label_s)

            for i, label in enumerate(label_s):
                indices.append([element, i])
                values.append(label)

        return indices, values, shape

    def train_batch(self, batch):
        # feed batch to nn and train
        n_batch_elements = len(batch.imgs)
        max_text_length = batch.imgs[0].shape[0] // 4
        sparse = self.to_sparse(batch.generated_texts)
        evaluate_list = [self.optimizer, self.loss]
        feed_dictionary = {self.input_imgs: batch.imgs, self.gt_texts: sparse,
                     self.seq_len: [max_text_length] * n_batch_elements, self.is_train: True}
        _, loss_val = self.sess.run(evaluate_list, feed_dictionary)
        self.batches_trained += 1
        return loss_val

    def output_text(self, ctc_output, batch_size):

    # Also known as infer_batch
    def recognize_text(self, batch, calc_probablity, text_probability):
        n_batch_elements = len(batch.imgs)
        evaluated_list = []

        evaluated_list.append(self.decoder)

        if self.dump or calc_probablity:
            evaluated_list.append(self.ctc_3_dimension)

        max_text_length = batch.imgs[0].shape[0] // 4

        # dictionary of tensors
        feed_dict = {self.input_images: batch.imgs, self.seq_len: [max_text_length] * n_batch_elements,
                     self.is_train: False}

        # evaluate model
        evaluated_result = self.sess.run(evaluated_list, feed_dict)

        decoded = evaluated_result[0]
        texts = self.output_text(decoded, n_batch_elements)

        # feed RNN output and recognized text to ctc loss for probability
        probs = None
        if calc_probablity:
            sparse = self.to_sparse(batch.generated_texts) if text_probability else self.to_sparse(texts)
            ctc_input = evaluated_result[1]
            evaluated_list = self.loss_per_element
            feed_dict = {self.saved_ctc_input: ctc_input, self.gt_texts: sparse,
                         self.seq_len: [max_text_length] * n_batch_elements, self.is_train: False}
            loss_values = self.sess.run(evaluated_list, feed_dict)
            probs = np.exp(-loss_values)

        return texts, probs

    def save_model(self):
        self.snap_ID += 1
        self.saver.save(self.sess, '../model/snapshot', self.snap_ID)