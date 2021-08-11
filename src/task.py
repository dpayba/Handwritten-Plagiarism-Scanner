import argparse
import json
from typing import Tuple, List
import os

import cv2
import editdistance
from path import Path

from load_data import LoadIAM, Batch
from model import Model
from preprocessor import Preprocessor


class Paths:
    char_list_path = '../model/charList.txt'
    summary_path = '../model/summary.json'
    corpus_path = '../data/corpus.txt'

def get_image_height():
    return 32

def get_image_size(line_mode: bool=False):
    if line_mode:
        return 256, get_image_height()
    return 128, get_image_height()

def validate(model, data_loader, line_mode):
    print('Validate Neural Network')
    data_loader.switch_to_validate()
    preprocessor = Preprocessor(get_image_size(line_mode), line_mode)
    n_char_error = 0
    n_char_total = 0
    n_words_match = 0
    n_words_total = 0
    while data_loader.has_next():
        iterator_info = data_loader.get_it_info()
        print(f'Batch: {iterator_info[0]} / {iterator_info[1]}')
        batch = data_loader.get_next()
        batch = preprocessor.process_batch(batch)
        recognized, _ = model.recognize_text(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            n_words_match += 1 if batch.generated_texts[i] == recognized[i] else 0
            n_words_total += 1
            distance = editdistance.eval(recognized[i], batch.generated_texts[i])
            n_char_error += distance
            n_char_total += len(batch.generated_texts[i])
            print('[OK]' if distance == 0 else '[ERR:%d]' % distance, '"' + batch.generated_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    char_error_rate = n_char_error / n_char_total
    word_accuracy = n_words_match / n_words_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy


def train(model, data_loader, line_mode, early_stopping):
    epoch = 0
    char_error_summary = []
    word_accuracy_summary = []
    preprocessor = Preprocessor(get_image_size(line_mode), data_augmentation=True, line_mode=line_mode)
    char_error_best = float('inf')
    n_epochs_nochange = 0

    while True:
        epoch += 1
        print('Epoch:', epoch)

        print('Train NN')
        data_loader.train_set()
        while data_loader.has_next():
            iterator_info = data_loader.get_it_info()
            batch = data_loader.get_next()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iterator_info[0]}/{iterator_info[1]} Loss: {loss}')


        # validate
        char_error_rate, word_accuracy = validate(model, data_loader, line_mode)

        # write summary
        char_error_summary.append(char_error_rate)
        word_accuracy_summary.append(word_accuracy)
        with open(Paths.summary_path, 'w') as f:
            json.dump({'charErrorRates': char_error_summary, 'wordAccuracies': word_accuracy_summary}, f)

        if char_error_rate < char_error_best:
            print('Error rate improved, save model')
            char_error_best = char_error_rate
            n_epochs_nochange = 0
            model.save_model()
        else:
            print('Character error rate not improved')
            n_epochs_nochange += 1

        if n_epochs_nochange >= early_stopping:
            print(f'No more improvement since {early_stopping} epochs. Training stopped')
            break

def infer(model, fn_image, output_dir):
    image = cv2.imread(fn_image, cv2.IMREAD_GRAYSCALE)
    assert image is not None

    preprocessor = Preprocessor(get_image_size(), width_dynamic=True, padding=16)
    image = preprocessor.process_image(image)

    batch = Batch([image], None, 1)
    recognized, probability = model.recognize_text(batch, True)
    print(f'Recognized: "{recognized[0]}"')
    print(f'Probability: {probability[0]}')

    parent_dir = os.path.dirname(output_dir)
    if not os.path.isdir(parent_dir):
        os.mkdir(parent_dir)
    file = open(output_dir, 'w')
    file.write(recognized[0])
    file.close()