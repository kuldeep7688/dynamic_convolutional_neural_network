# importing all the required libraries
import os
import sys
import torch
import pandas as pd
import torchtext
import torch.nn as nn
from torchtext import data
import torch.optim as optim
from datetime import datetime
from os.path import abspath
from argparse import ArgumentParser

# importing local modules
script_path = os.path.abspath('')
sys.path.insert(0, abspath(script_path))
# print(abspath(script_path))

from utils.utilities import load_check_point,\
    tokenizer_nltk, load_dict_from_disk

# importing model
from model.model import DCNN_SST

# importing the model parameters
from utils.model_parameters import SST1_DATASET_PARAMETERS

parser = ArgumentParser()
parser.add_argument(
    "--embedding_dim", help="Mention the dimension of embedding.",
    type=int,
    default=300
)
parser.add_argument(
    "--sentence_length", help="Fix the sentence length for each sentence.",
    type=int,
    default=18
)
parser.add_argument(
    "--saved_model_path", help="Mention the path where model is saved.",
    type=str,
    default=None
)
parser.add_argument(
    "--saved_vocab_path", help="Mention the path where vocab is saved.",
    type=str,
    default=None
)
parser.add_argument(
    "--device", help="Mention the device to be used cuda or cpu,",
    type=str,
    default=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)
parser.add_argument(
    "--glove_file_path", help="Mention the path where glove embeddings are saved.",
    type=str,
    default="/home/neo/glove.6B.300d.txt"
)
parser.add_argument(
    "--file_to_predict_on", help="Mention the path of the csv file to predict on.",
    type=str,
    default=None
)
parser.add_argument(
    "--file_to_save_predictions", help="Mention the path of the csv file to save predictions.",
    type=str,
    default=None
)
arguments = parser.parse_args()
EMBEDDING_DIM = arguments.embedding_dim
SST1_DATASET_PARAMETERS["embedding_dim"] = EMBEDDING_DIM

SENT_LENGTH = arguments.sentence_length
SST1_DATASET_PARAMETERS["cell_one_parameter_dict"]["sent_length"] = SENT_LENGTH

MODEL_PATH = arguments.saved_model_path
VOCAB_PATH = arguments.saved_vocab_path
SAVE_PATH = arguments.file_to_save_predictions
DEVICE = arguments.device

FILE_TO_PREDICT_ON = arguments.file_to_predict_on

GLOVE_FILE_PATH = arguments.glove_file_path


def return_indexed(vocab_obj, tokenized):
    indexed = []
    for i in tokenized:
        if len(i) < SENT_LENGTH:
            i = i + ["<pad>"]*(SENT_LENGTH - len(i))

        if len(i) > SENT_LENGTH:
            i = i[:SENT_LENGTH]
        temp = []
        for j in i:
            temp.append(vocab_obj.stoi[j])
        indexed.append(temp)
    return indexed



def predict_using_model(model, vocab):
    df = pd.read_csv(FILE_TO_PREDICT_ON)
    df["sentence_tokenized"] = df["sentence"].apply(lambda x: tokenizer_nltk(x))
    df["indexed"] = return_indexed(vocab, df["sentence_tokenized"])

    input_tensor = torch.LongTensor(list(df["indexed"])).to(DEVICE)
    model_outputs = model(input_tensor).squeeze(1)
    preds, ind = torch.max(torch.nn.functional.softmax(model_outputs, dim=-1), 1)
    preds = preds.cpu().detach().numpy()
    ind = ind.cpu().detach().numpy()

    df["predictions"] = ind
    df["probabilities"] = preds
    df = df[["sentence", "predictions", "probabilities"]]
    df.to_csv(SAVE_PATH, index=False, encoding="utf-8")
    return


if __name__ == "__main__":

    vocab = load_dict_from_disk(VOCAB_PATH)
    SST1_DATASET_PARAMETERS["vocab_length"] = len(vocab.stoi)

    model = DCNN_SST(parameter_dict=SST1_DATASET_PARAMETERS)

    model.to(DEVICE)

    model = load_check_point(model, MODEL_PATH)
    predict_using_model(model, vocab)

    print("\n\n")
    print("FINISH")
    print("############################################################################")

