from __future__ import print_function
from hyperparams import Hyperparams as hp
from MLM import*
import tensorflow as tf
import numpy as np
import codecs
import regex
import sys

def load_de_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de_en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_tw_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/tw.vocab.tsv', 'r', 'utf-8').read().splitlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents):
    token2idx, idx2token = load_de_en_vocab()
    turn_over_count = 0

    # Index
    x_list, y_list, y_decoder_input_list, Sources, Targets = [], [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [token2idx.get(word, 1) for word in (source_sent).split()]
        y = [token2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
        y_decoder_input = [token2idx.get(word, 1) for word in target_sent.split()]

        if (len(y) <= hp.maxlen) and (len(x) <= hp.maxlen):
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            y_decoder_input_list.append(np.array(y_decoder_input))
            Sources.append(source_sent)
            Targets.append(target_sent)
        else:
            turn_over_count += 1
            print("maxlen!%d",turn_over_count)
            #sys.exit(1)

    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    Y_DI = np.zeros([len(y_decoder_input_list), hp.maxlen], np.int32)

    for i, (x, y, y_decoder_input) in enumerate(zip(x_list, y_list, y_decoder_input_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
        Y_DI[i] = np.lib.pad(y_decoder_input, [0, hp.maxlen-len(y_decoder_input)], 'constant', constant_values=(0, 0))

    return X, Y, Y_DI, Sources, Targets

def load_train_data():
    de_sents = [line for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line]
    en_sents = [line for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line]
    X, Y, Y_DI, Sources, Targets = create_data(de_sents, en_sents)
    return X, Y, Y_DI, Sources, Targets

def load_test_data():
    de_sents = [line for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if line]
    en_sents = [line for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if line]
    X, Y, Y_DI, Sources, Targets = create_data(de_sents, en_sents)
    return X, Y, Sources, Targets

def get_batch_data():
    # Load data
    X, Y, Y_DI, Sources, Targets = load_train_data()

    # calc total batch count
    num_batch = len(X) // hp.batch_size

    return X, Y, Y_DI, num_batch     # (N, T), (N, T), ()

