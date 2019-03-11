import numpy as np
import pandas as pd
import pyprind
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import torch
import torch.optim as optim
from torchtext import data
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from pprint import pprint
import math
import nltk
import pyprind
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import sys
import os
from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import utils
from utils.pytorchtext_adv import _DataLoaderIterAdv, DataLoader
from utils.utilities import VectorizeData, load_glove_model, create_embedding_matrix, create_word2idx, evaluate, train, save_checkpoint, load_check_point
from utils.utilities import calculate_accuracy, print_number_of_trainable_parameters

from utils.model_parameters import SST2_DATASET_PARAMETERS
pprint(SST2_DATASET_PARAMETERS)

# importing model
import model
from model.model import DCNNCell, DCNN_SST2, Flatten 

if __name__ == "__main__":
    glove_file_path = "/home/neo/glove.6B.100d.txt"
    glove2embedding = load_glove_model(glove_file_path)
    
    data_file_path = "/home/neo/sentiment_datasets/SST2.csv"
    data = pd.read_csv(data_file_path)
    print(data.head())
    print()
    print(f"Shape of the data is {data.shape}")

    word2idx = create_word2idx(data["sentence"])
    idx2word = {idx: word for word, idx in word2idx.items()}

    SST2_DATASET_PARAMETERS["vocab_length"] = len(word2idx)

    embedding_weights_matrix = create_embedding_matrix(
        embedding_dim=SST2_DATASET_PARAMETERS["embedding_dim"],
        word2idx=word2idx,
        embedding_function=glove2embedding)

    print()
    print(f"Shape of the embeddings weights matrix is {embedding_weights_matrix.shape}")

    df_train = data.loc[data.split == "train", :].reset_index(drop=True)
    df_dev = data.loc[data.split == "dev", : ].reset_index(drop=True)
    df_test = data.loc[data.split == "test", :].reset_index(drop=True)

    print()
    print(f"Shape of the training data is {df_train.shape}")
    print(f"Shape of the development data is {df_dev.shape}")
    print(f"Shape of the test data is {df_test.shape}")
    
    maxlen=20
    ds_train = VectorizeData(
        df=df_train,
        maxlen=maxlen,
        word2idx=word2idx,
        text_column_name="sentence",
        label_column_name="label",
        constant_sent_length=False,
        prepare_batchces_maxlen=True
    )

    ds_dev = VectorizeData(
        df=df_dev,
        maxlen=maxlen,
        word2idx=word2idx,
        text_column_name="sentence",
        label_column_name="label",
        constant_sent_length=False,
        prepare_batchces_maxlen=True
    )


    ds_test = VectorizeData(
        df=df_test,
        maxlen=maxlen,
        word2idx=word2idx,
        text_column_name="sentence",
        label_column_name="label",
        constant_sent_length=False,
        prepare_batchces_maxlen=True
    )
    batch_size = 20
    dl_train = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
    dl_dev = DataLoader(dataset=ds_dev, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=True)

    print()
    print(f"Total batches in train data loader are {len(dl_train)}")
    print(f"Total batches in test data loader are {len(dl_test)}")
    print(f"Total batches in dev data loader are {len(dl_dev)}")

    for i, j in enumerate(dl_train):
        xs, ys = j
        print(xs.shape)
        if i == 20:
            break

    print()
    print("Loading DCNN Model....")
    model = DCNN_SST2(
        parameter_dict=SST2_DATASET_PARAMETERS
    )
    model.embedding.weight.data.copy_(torch.from_numpy(embedding_weights_matrix))

    print_number_of_trainable_parameters(model)

    optimizer = optim.Adam(model.parameters(), weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print()
    print("Device selected by the script is {device}")
    model = model.to(device)
    criterion = criterion.to(device)
    MODEL_PATH = "/home/neo/github_projects/dynamic_convolutional_neural_network/data/"\
            "pretrained_models/{}_sst2_model.tar".format(datetime.today().strftime('%Y-%m-%d'))
    n_epochs = 10
    base_dev_acc = 0.0
    for epoch in range(n_epochs):
        is_best = False
        train_loss, train_acc = train(model, dl_train, optimizer, criterion, device=device)
        valid_loss, valid_acc = evaluate(model, dl_dev, criterion, device=device)
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')
        print()
        if base_dev_acc < valid_acc:
            is_best = True
            base_dev_acc = valid_acc

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_loss": valid_loss,
                "best_dev_accuracy": valid_acc,
            },
            is_best,
            MODEL_PATH
        )
