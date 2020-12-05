import argparse
import time
import torch
import os
from os.path import join, exists
from collections import Counter
import numpy as np
import pickle
from preprocessing import init_vocab
from model import CharLM
from tqdm import tqdm

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
else:
    torch.manual_seed(42)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'sample'], default='train')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='path to parent data directory')
    parser.add_argument('--dump-dir', type=str, required=True,
                        help='path to dump directory')
    parser.add_argument('--model-file', type=str,
                        help='path to model file')

    # training details
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of lstm layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout rate')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='label smoothing rate')
    parser.add_argument('--embed-dim', type=int, default=128,
                        help='hidden size')
    parser.add_argument('--epoch-size', type=int, default=500,
                        help='number of training steps')
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='number of epochs')
    # Karpathy's Constant, lol
    # https://twitter.com/karpathy/status/801621764144971776?lang=en
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Adam learning rate')

    return parser



def get_data(data_dir, mode):
    input_file = join(data_dir, '{}.input.npy'.format(mode))
    target_file = join(data_dir, '{}.target.npy'.format(mode))
    inp = np.load(input_file, allow_pickle=True)
    target = np.load(target_file, allow_pickle=True)
    return inp, target

def train(args, model):
    num_epochs = args.num_epochs
    epoch_size = args.epoch_size
    data_dir = args.data_dir
    dump_dir = args.dump_dir
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_in, train_out = get_data(data_dir, 'train')
    dev_in, dev_out = get_data(data_dir, 'dev')
    all_dev_ppls = []

    for e_idx in tqdm(range(num_epochs)):
        epoch_loss = 0.
        epoch_nll_loss = 0.
        epoch_weight = 0.

        # train
        start = time.time()
        for _ in tqdm(range(epoch_size)):
            # sample a batch
            idxs = np.random.randint(low=0, high=train_in.shape[0], size=(32))
            inp = torch.from_numpy(train_in[idxs]).to(device)
            tar = torch.from_numpy(train_out[idxs]).to(device)
            ret = model(inp, tar)
            opt_loss = ret['opt_loss']
            loss = ret['loss'].item()
            nll_loss = ret['nll_loss'].item()
            num_words = ret['num_words']

            # update
            optimizer.zero_grad()
            opt_loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # update stats
            epoch_loss += loss
            epoch_nll_loss += nll_loss
            epoch_weight += num_words

        # finish one epoch
        print('Finish epoch {}, it takes {} seconds'.format(e_idx + 1, time.time() - start))
        wps = epoch_weight / (time.time() - start)
        smppl = epoch_loss / epoch_weight
        smppl = np.exp(smppl) if smppl < 300 else 1e9
        ppl = epoch_nll_loss / epoch_weight
        ppl = np.exp(ppl) if ppl < 300 else 1e9
        print('train_smppl = {:.3f}, train_ppl = {:.3f}, wps={} toks/s'.format(smppl, ppl, wps))
        epoch_loss = 0.
        epoch_nll_loss = 0.
        epoch_weight = 0.

        # eval & save checkpoint
        print('Evaluate on dev')
        model.eval()
        dev_size = dev_in.shape[0]
        dev_loss = 0.
        dev_nll_loss = 0.
        dev_weight = 0.
        for i in tqdm(range(0, dev_size, 32)):
            j = min(i + 32, dev_size)
            inp = torch.from_numpy(dev_in[i:j]).to(device)
            tar = torch.from_numpy(dev_out[i:j]).to(device)
            ret = model(inp, tar)
            loss = ret['loss'].item()
            nll_loss = ret['nll_loss'].item()
            num_words = ret['num_words']

            dev_loss += loss
            dev_nll_loss += nll_loss
            dev_weight += num_words

        smppl = dev_loss / dev_weight
        smppl = np.exp(smppl) if smppl < 300 else 1e9
        ppl = dev_nll_loss / dev_weight
        ppl = np.exp(ppl) if ppl < 300 else 1e9
        print('dev_smppl = {:.3f}, dev_ppl = {:.3f}'.format(smppl, ppl))
        all_dev_ppls.append(ppl)
        if all_dev_ppls[-1] == min(all_dev_ppls):
            # save checkpoint
            ckpt_path = join(dump_dir, 'model-{}.pth'.format(all_dev_ppls[-1]))
            torch.save(model.state_dict(), ckpt_path)

            # remove old checkpoint
            if len(all_dev_ppls) > 1 and min(all_dev_ppls[:-1]) != all_dev_ppls[-1]:
                ckpt_path = join(dump_dir, 'model-{}.pth'.format(min(all_dev_ppls[:-1])))
                if exists(ckpt_path):
                    os.remove(ckpt_path)

        model.train()


if __name__ == '__main__':
    args = get_parser().parse_args()
    data_dir = args.data_dir
    vocab_file = join(data_dir, 'vocab.pkl')
    vocab, ivocab = init_vocab(vocab_file)
    args.vocab_size = len(vocab)

    model = CharLM(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'sample':
        model_file = args.model_file
        model.load_state_dict(torch.load(model_file))
        seed = "Đoàn Dự nói:"
        seed = [vocab.get(ch, 0) for ch in seed]
        seed = torch.tensor(seed).to(device).reshape(1, -1)
        max_length = 10000
        output = model.sample(seed, max_length=max_length)
        output = [ivocab.get(idx) for idx in output]
        print('Given the seed "{}", model produces the following:'.format(seed))
        print(''.join(output))
