import argparse
from os.path import join, exists
from collections import Counter
import numpy as np
import pickle

UNK_ID = 0
UNK = "_UNK_"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preprocess', 'process'], default='preprocess')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='path to parent data directory')
    parser.add_argument('--test-file', type=str,
                        help='convert this file into npy')
    return parser


def init_vocab(vocab_file):
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)

    ivocab = {v: k for k, v in vocab.items()}
    return vocab, ivocab


def extract_vocab(train_file, vocab_file):
    if exists(vocab_file):
        print('Vocab file {} exists.'.format(vocab_file))
        return

    # extract character vocab
    with open(train_file) as f:
        data = f.read()

    vocab = Counter(data)
    items = vocab.most_common()
    d = {UNK: UNK_ID}
    i = UNK_ID
    for (ch, count) in items:
        i += 1
        d[ch] = i

    with open(vocab_file, 'wb') as fout:
        pickle.dump(d, fout)


def convert2npy(data_file, input_npy_file, target_npy_file, vocab):
    if exists(input_npy_file) and exists(target_npy_file):
        print('{} & {} exist.'.format(input_npy_file, target_npy_file))
        return

    with open(data_file) as f:
        data = f.read()

    data = [vocab.get(ch, UNK_ID) for ch in data]
    inputs = []
    targets = []
    CHUNK = 128
    for i in range(0, len(data) - 1, CHUNK):
        j = i + CHUNK
        if j + 1 < len(data):
            x = data[i:j]
            y = data[i+1:j+1]
            inputs.append(x)
            targets.append(y)

    inputs = np.array(inputs)
    targets = np.array(targets)
    np.save(input_npy_file, inputs, allow_pickle=True)
    np.save(target_npy_file, targets, allow_pickle=True)


def preprocess(args):
    data_dir = args.data_dir
    train_file = join(data_dir, 'train.txt')
    dev_file = join(data_dir, 'dev.txt')
    vocab_file = join(data_dir, 'vocab.pkl')

    # get vocab file
    extract_vocab(train_file, vocab_file)
    vocab, ivocab = init_vocab(vocab_file)

    # convert data into npy
    for mode in ['train', 'dev']:
        data_file = train_file if mode == 'train' else dev_file
        input_npy_file = join(data_dir, '{}.input.npy'.format(mode))
        target_npy_file = join(data_dir, '{}.target.npy'.format(mode))
        convert2npy(data_file, input_npy_file, target_npy_file, vocab)

def process(args):
    # process test for evaluation, probably not gonna use this
    vocab_file = join(data_dir, 'vocab.pkl')
    vocab, ivocab = init_vocab(vocab_file)
    test_file = args.test_file
    with open(test_file) as f:
        data = f.read()

    data = [vocab.get(ch, UNK_ID) for ch in data]
    npy_file = test_file + '.npy'
    np.save(npy_file, np.array(data), allow_pickle=True)


if __name__ == '__main__':
    args = get_parser().parse_args()
    if args.mode == 'preprocess':
        preprocess(args)
    elif args.mode == 'process':
        process(args)
    else:
        raise ValueError('Mode {} is not supported!'.format(args.mode))
