import argparse
import json
from typing import Tuple, List

import cv2
from path import Path

from load_data import LoadIAM, Batch
from model import Model
from preprocessor import Preprocessor

def get_image_height():
    return 32

def get_image_size(line_mode):
    if line_mode:
        return 256, get_image_height()
    return 128, get_image_height()

def validate(model, data_loader, line_mode):
    print('Validate Neural Network')

def train(model, data_loader, line_mode, early_stopping):
    epoch = 0
    char_error_summary = []
    word_accuracy_summary = []
    preprocessor = Preprocessor(get_image_size(line_mode), True, line_mode)
    char_error_best = float('inf')
    n_epochs_nochange = 0

    while True:
        epoch += 1
        print('Epoch:', epoch)

        print('Train NN')
        data_loader.train_set()
        while data_loader.has_next():
            iterator_info = data_loader.get_it_info
            batch = data_loader.get_next()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iterator_info[0]}/{iterator_info[1]} Loss: {loss}')


        # validate
        char_error_rate, word_accuracy = validate(model, data_loader, line_mode)