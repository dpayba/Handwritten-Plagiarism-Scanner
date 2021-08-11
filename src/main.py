import argparse
import json
from typing import Tuple, List

import cv2
from path import Path

from load_data import LoadIAM, Batch
from model import Model
import task as ml
from task import Paths

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--batch_size', help='Batch Size', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory of IAM dataset', type=Path, required=False)
    parser.add_argument('--fast', help='Load from LMDB', action='store_true')
    parser.add_argument('--line_mode', help='Train to read text lines instead of words', action='store_true')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../data/word.png')

    args = parser.parse_args()
    # choose train or validate IAM
    if args.mode in ['train', 'validate']:
        data_loader = LoadIAM(args.data_dir, args.batch_size)
        char_list = data_loader.char_list

        if args.line_mode and ' ' not in char_list:
            char_list = [' '] + char_list

        open(Paths.char_list_path, 'w').write(''.join(char_list))

        open(Paths.corpus_path, 'w').write(' '.join(data_loader.train_words + data_loader.validation_words))

        if args.mode == 'train':
            model = Model(char_list)
            ml.train(model, data_loader, args.line_mode, args.early_stopping)
        elif args.mode == 'validate':
            model = Model(char_list, must_restore=True)
            ml.validate(model, data_loader, args.line_mode)
    elif args.mode == 'infer':
        model = Model(list(open(Paths.char_list_path).read()), must_restore=True)
        ml.infer(model, args.img_file)

if __name__ == '__main__':
    main()