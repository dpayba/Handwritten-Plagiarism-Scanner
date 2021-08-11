import os
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from load_data import Batch

tf.compat.v1.disable_eager_execution()

class Model:
    def __init__(self, char_list: List[str],
                 must_restore: bool = False):

        self.char_list = char_list
        self.must_restore = must_restore
        self.snap_ID = 0

        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

        self.input_images = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None))

        self.create_cnn()
        self.create_rnn()
        self.create_ctc()

        self.batches_trained = 0
        self.update_operations = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_operations):
            self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)

        self.sess, self.saver = self.setup_tf()

    def create_cnn(self):
        cnn_4_dimension = tf.expand_dims(input=self.input_images, axis=3)

        # layer parameters
        kernel_vals = [5, 5, 3, 3, 3]
        features = [1, 32, 64, 128, 128, 256]
        strides = pool_vals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        n_layers = len(strides)

        pool = cnn_4_dimension
        for i in range(n_layers):
            kernel = tf.Variable(tf.random.truncated_normal([kernel_vals[i], kernel_vals[i], features[i], features[i+1]],
                                                            stddev=0.1))
            conv = tf.nn.conv2d(input=pool, filters=kernel, padding='SAME', strides=(1, 1, 1, 1))
            conv_normalize = tf.compat.v1.layers.batch_normalization(conv, training=self.is_train)
            relu = tf.nn.relu(conv_normalize)
            pool = tf.nn.max_pool2d(input=relu, ksize=(1, pool_vals[i][0], pool_vals[i][1], 1),
                                    strides=(1, strides[i][0], strides[i][1], 1), padding='VALID')

        self.cnn_output = pool

    def create_rnn(self):
        rnn_3_dimension = tf.squeeze(self.cnn_output, axis=[2])
        n_hidden = 256
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=n_hidden, state_is_tuple=True) for _ in
                 range(2)]  # 2 layers

        # stack cells basic
        stack_cells = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional rnn
        (fw, bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=stack_cells, cell_bw=stack_cells, inputs=rnn_3_dimension,
                                                                dtype=rnn_3_dimension.dtype)
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        kernel = tf.Variable(tf.random.truncated_normal([1, 1, n_hidden*2, len(self.char_list) + 1], stddev=0.1))
        self.rnn_output = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'),
                                     axis=[2])

    def create_ctc(self):
        self.ctc_3_dimension = tf.transpose(self.rnn_output, [1, 0, 2])

        self.generated_texts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[None, 2]),
                                        tf.compat.v1.placeholder(tf.int32, [None]),
                                        tf.compat.v1.placeholder(tf.int64, [2]))

        self.seq_len = tf.compat.v1.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(input_tensor=tf.compat.v1.nn.ctc_loss(labels=self.generated_texts, inputs=self.ctc_3_dimension,
                                                  sequence_length=self.seq_len,
                                                  ctc_merge_repeated=True))

        self.saved_ctc_input = tf.compat.v1.placeholder(tf.float32, [None, None, len(self.char_list) + 1])
        self.loss_per_element = tf.compat.v1.nn.ctc_loss(self.generated_texts, self.saved_ctc_input,
                                                         self.seq_len, True)
        self.decoder = tf.nn.ctc_greedy_decoder(self.ctc_3_dimension, self.seq_len)

    def setup_tf(self):
        sess = tf.compat.v1.Session()

        saver = tf.compat.v1.train.Saver(max_to_keep=1)
        model_dir = '../model/'
        latest_snapshot = tf.train.latest_checkpoint(model_dir)

        if self.must_restore and not latest_snapshot:
            raise Exception('No saved model found in: ' + model_dir)

        if latest_snapshot:
            print('Initialize with stored values from ' + latest_snapshot)
            saver.restore(sess, latest_snapshot)
        else:
            print('Initialize with new values')
            sess.run(tf.compat.v1.global_variables_initializer())

        return sess, saver

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
        feed_dictionary = {self.input_images: batch.imgs, self.generated_texts: sparse,
                     self.seq_len: [max_text_length] * n_batch_elements, self.is_train: True}
        _, loss_val = self.sess.run(evaluate_list, feed_dictionary)
        self.batches_trained += 1
        return loss_val

    def output_text(self, ctc_output, batch_size):
        decoded = ctc_output[0][0]
        label_s = [[] for _ in range(batch_size)]

        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batch_element = idx2d[0]
            label_s[batch_element].append(label)

        return [''.join([self.char_list[c] for c in labelStr]) for labelStr in label_s]

    """ Also known as infer_batch"""
    def recognize_text(self, batch: Batch, calc_probablity: bool=False, text_probability: bool=False):
        n_batch_elements = len(batch.imgs)
        evaluated_list = []

        evaluated_list.append(self.decoder)

        if calc_probablity:
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
            feed_dict = {self.saved_ctc_input: ctc_input, self.generated_texts: sparse,
                         self.seq_len: [max_text_length] * n_batch_elements, self.is_train: False}
            loss_values = self.sess.run(evaluated_list, feed_dict)
            probs = np.exp(-loss_values)

        return texts, probs

    def save_model(self):
        self.snap_ID += 1
        self.saver.save(self.sess, '../model/snapshot', self.snap_ID)