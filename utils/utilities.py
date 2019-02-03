
import math
import pandas as pd
import numpy as np
from datetime import datetime
import pyprind
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
tqdm.pandas()

def load_glove_model(embedding_file_path):
    word2idx = {}
    glove2embedding = {}
    bar_count = len(open(embedding_file_path, 'rb').readlines())
    with open(embedding_file_path, 'rb') as f:
        bar = pyprind.ProgBar(bar_count, bar_char='█')
        for idx, l in enumerate(f):
            line = l.decode().split()
            word = line[0]
#             words.append(word)
            word2idx[word] = idx

            vect = np.array(line[1:]).astype(np.float)
            glove2embedding[word] = vect
            idx += 1
            bar.update()
    print("Total vocabulary size of Embedding model is {}.".format(len(word2idx)))
    return glove2embedding


def create_word2idx(list_of_text):
    words = {"_pad_": 0}
    idx = 1
    bar = pyprind.ProgBar(len(list_of_text), bar_char='█')
    for i in list_of_text:
        for w in nltk.word_tokenize(i):
            if w.lower() not in words.keys():
                words[w.lower()] = idx
                idx += 1
        bar.update()
    print(f'Number of unique tokens in the data are {len(words)}')
    return words


def create_embedding_matrix(embedding_dim, word2idx, embedding_function):
    matrix_len = len(list(word2idx.keys()))
    weights_matrix = np.zeros((matrix_len, embedding_dim), dtype=float)
    words_found = 0
    bar = pyprind.ProgBar(len(word2idx), bar_char='█')
    for i, word in enumerate(list(word2idx.keys())):
        try: 
            weights_matrix[i] = embedding_function[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(
                scale=0.6, size=(embedding_dim,)
            )
        bar.update()
    print(f"Number of words from text found in embedding function are {words_found}")
    return weights_matrix


class VectorizeData_SST(Dataset):
    def __init__(self, df=None, maxlen=10, word2idx=None):
        self.maxlen = maxlen
        self.df = df
        self.df['sentence'] = self.df.sentence.progress_apply(
            lambda x: x.strip()
        )
        print('Indexing...')
        self.df['sentimentidx'] = self. df.sentence.progress_apply(
            lambda x: [
                word2idx[w.lower()]
                for w in nltk.word_tokenize(x.strip())
            ]
        )
        print('Calculating lengths')
        self.df['lengths'] = self.df.sentimentidx.progress_apply(
            lambda x: self.maxlen if len(x) > self.maxlen else len(x)
        )
        print('Padding')
        self.df['sentimentpadded'] = self.df.sentimentidx.progress_apply(
            self.pad_data
        )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        X = self.df.sentimentpadded[idx]
        y = self.df.label[idx]
        return X, y

    def pad_data(self, s):
        padded = np.zeros((self.maxlen,), dtype=np.int64)
        if len(s) > self.maxlen: padded[:] = s[:self.maxlen]
        else: padded[:len(s)] = s
        return padded


def print_number_of_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable parameters in the model are : {}".format(params))
    return


def calculate_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    preds, ind= torch.max(F.softmax(preds, dim=-1), 1)
    correct = (ind == y).float()
    acc = correct.sum()/float(len(correct))
    return acc


def train(model, iterator, optimizer, criterion, device="cpu"):

    epoch_loss = 0
    epoch_acc = 0

    model.train()
    bar = pyprind.ProgBar(len(iterator), bar_char='█')

    for i, batch in enumerate(iterator):
        inputs, labels = batch
        if device == "cpu":
            x, y = Variable(inputs), Variable(labels.long())
        else:
            x, y = Variable(inputs.cuda()), Variable(labels.long().cuda())

        optimizer.zero_grad()
        predictions = model(x).squeeze(1)
        loss = criterion(predictions, y)
        acc = calculate_accuracy(predictions, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        bar.update()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device="cpu"):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        bar = pyprind.ProgBar(len(iterator), bar_char='█')
        for i, batch in enumerate(iterator):
            inputs, labels = batch
            if device == "cpu":
                x, y = Variable(inputs), Variable(labels.long())
            else:
                x, y = Variable(inputs.cuda()), Variable(labels.long().cuda())

            predictions = model(x).squeeze(1)
            loss = criterion(predictions, y)
            acc = calculate_accuracy(predictions, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            bar.update()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def save_checkpoint(state, is_best, filename):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation loss did not improve")
    return


def load_check_point(model_path):
    resume_weights = model_path
    checkpoint = torch.load(resume_weights)
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_dev_accuracy']
    model.load_state_dict(checkpoint['state_dict'])
    print("Best Dev Accuracy is {}".format(best_accuracy))
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights, checkpoint['epoch']))
    return model
