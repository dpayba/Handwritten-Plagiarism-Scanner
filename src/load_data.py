import pickle
import random
from collections import namedtuple
from typing import Tuple

# Data
import cv2
import lmdb
import numpy as np
from path import Path

Sample = namedtuple('Sample', 'generated_text, file_path')
Batch = namedtuple('Batch', 'imgs, generated_texts, batch_size')

class LoadIAM:
    def __init__(self, data_dir, batch_size, data_split=0.90):
        assert(data_dir.exists())

        fast = True
        self.fast = fast
        if fast:
            self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)

        self.data_augmentation = False
        self.cur_index = 0
        self.batch_size = batch_size
        self.samples = []

        f = open(data_dir / 'get_text/words.txt')
        chars = set()
        # filter bad images
        bad_images = ['a01-117-05-02', 'r06-022-03-05']
        for line in f:
            if not line or line[0] == '#':
                continue

            line_split = line.strip().split('')
            assert len(line_split) >= 9

            file_name_split = line_split[0].split('-')
            file_name_subdirectory1 = file_name_split[0]
            file_name_subdirectory2 = file_name_split[0]
            file_base_name = line_split[0] + '.png'
            file_name = data_dir / 'img' / file_name_subdirectory1 / file_name_subdirectory2 / file_base_name

            if line_split[0] in bad_images:
                print('Ignore image', file_name)
                continue

            # generated text
            generated_text = ' '.join(line_split[8:])
            chars = chars.union(set(list(generated_text)))

            # add sample to list
            self.samples.append(Sample(generated_text, file_name))

        # split train and validate sets, start with 10/90
        split_index = int(data_split * len(self.samples))
        self.train_samples = self.samples[:split_index]
        self.validation_samples = self.samples[split_index:]

        # put words to lists
        self.train_words = [x.gt_text for x in self.train_samples]
        self.validation_words = [x.gt_text for x in self.validation_samples]

        self.train_set()

        self.char_list = sorted(list(chars))

    def train_set(self):
        self.data_augmentation = True
        self.current_index = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.current_set = 'train'

    def has_next(self):
        if self.current_set == 'train':
            return self.current_index + self.batch_size <= len(self.samples)
        else:
            return self.current_index < len(self.samples)

    def get_next(self):
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]

        self.curr_idx += self.batch_size
        return Batch(imgs, gt_texts, len(imgs))

    def get_it_info(self):
        if self.current_set == 'train':
            n_batches = int(np.floor(len(self.samples) / self.batch_size))
        else:
            n_batches = int(np.ceil(len(self.samples) / self.batch_size))
        current_batch = self.current_index // self.batch_size + 1
        return current_batch, n_batches