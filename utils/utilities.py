import torch
import nltk
import torch.nn.functional as F
import numpy as np
import pyprind


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
        x, y = batch

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
            x, y = batch

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


def load_check_point(model, model_path):
    resume_weights = model_path
    checkpoint = torch.load(resume_weights)
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_dev_accuracy']
    model.load_state_dict(checkpoint['state_dict'])
    print("Best Dev Accuracy is {}".format(best_accuracy))
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights, checkpoint['epoch']))
    return model

# tokenizer to be used in Field of torchtext
def tokenizer_nltk(x):
    return nltk.word_tokenize(x.lower())