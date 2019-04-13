# importing all the required libraries
import os
import sys
import torch
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

from utils.utilities import print_number_of_trainable_parameters\
    , load_check_point, tokenizer_nltk, train, evaluate, \
    calculate_accuracy, save_checkpoint

# importing model
from model.model import DCNN_SST

# importing the model parameters
from utils.model_parameters import SST2_DATASET_PARAMETERS

parser = ArgumentParser()
parser.add_argument(
    "--embedding_dim", help="Mention the dimension of embedding.",
    type=int,
    default=300
)
parser.add_argument(
    "--sentence_length", help="Fix the sentence length for each sentence.",
    type=int,
    default=19
)
parser.add_argument(
    "--save_path_for_model", help="Mention the path for saving the model.",
    type=str,
    default="data/trained_models/trained_sst2_model_{}.tar".format(str(datetime.now()).replace(" ", "_"))
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
    "--csv_folder_path", help="Mention the folder path where train, test and validation csv files are stored.",
    type=str,
    default="data/sst2_data/"
)
parser.add_argument(
    "--train_file_name", help="Mention the train csv file name.",
    type=str,
    default="sst2_train.csv"
)
parser.add_argument(
    "--val_file_name", help="Mention the validation csv file name",
    type=str,
    default="sst2_dev.csv"
)
parser.add_argument(
    "--test_file_name", help="Mention the test csv file name.",
    type=str,
    default="sst2_test.csv"
)
parser.add_argument(
    "--max_vocab_size", help="Maximum size of the vocab.",
    type=int,
    default=20000
)
parser.add_argument(
    "--train_batch_size", help="Mention the batch size for training the data.",
    type=int,
    default=32
)
parser.add_argument(
    "--val_batch_size", help="Mention the batch size for validation the data.",
    type=int,
    default=32
)
parser.add_argument(
    "--epochs", help="Mention the number of epochs to train the data on.",
    type=int,
    default=1
)
parser.add_argument(
    "--field_to_be_sorted_on", help="Mention the name of the fiels in csv which should be used for sorting.",
    type=str,
    default="sentence"
)
arguments = parser.parse_args()
EMBEDDING_DIM = arguments.embedding_dim
SST2_DATASET_PARAMETERS["embedding_dim"] = EMBEDDING_DIM

SENT_LENGTH = arguments.sentence_length
SST2_DATASET_PARAMETERS["cell_one_parameter_dict"]["sent_length"] = SENT_LENGTH

MODEL_PATH = arguments.save_path_for_model
DEVICE = arguments.device

GLOVE_FILE_PATH = arguments.glove_file_path
CSV_FOLDER_PATH = arguments.csv_folder_path
TRAIN_FILE_NAME = arguments.train_file_name
VALIDATION_FILE_NAME = arguments.val_file_name
TEST_FILE_NAME = arguments.test_file_name
MAX_VOCAB_SIZE = arguments.max_vocab_size
BATCH_SIZE_TRAIN = arguments.train_batch_size
BATCH_SIZE_VALIDATION = arguments.val_batch_size
BATCH_SIZE_TEST = arguments.val_batch_size
FIELD_TO_BE_SORTED_ON = arguments.field_to_be_sorted_on

CSV_FIELD_NAMES = [
    "sentence",
    "label",
    "split"
]

N_EPOCHS = arguments.epochs

def prepare_data_loaders(
        sent_length,
        tokenizer_func,
        csv_field_names,
        csv_folder_path,
        train_file_name,
        validation_file_name,
        test_file_name,
        glove_file_path,
        max_vocab_size,
        batch_size_train,
        batch_size_validation,
        batch_size_test,
        field_to_be_sorted_on,
        device
):
    text_field = data.Field(
        sequential=True,
        use_vocab=True,
        #     init_token="<ios>",
        #     eos_token="<eos>",
        fix_length=sent_length,
        tokenize=tokenizer_func,
        batch_first=True
    )
    label_field = data.Field(
        sequential=False,
        use_vocab=False,
        is_target=True
    )
    csv_fields = [
        (csv_field_names[0], text_field),
        (csv_field_names[1], label_field),
        (csv_field_names[2], None)
    ]

    trainds, valds, testds = data.TabularDataset.splits(
        path=csv_folder_path,
        format="csv",
        train=train_file_name,
        validation=validation_file_name,
        test=test_file_name,
        fields=csv_fields,
        skip_header=True
    )
    print(
        "length of train, validation and test data are respectively {}, {} and {}".format(
             len(trainds), len(valds), len(testds)
        )
    )
    print("Loading vectors from the file....")
    vec = torchtext.vocab.Vectors(glove_file_path)
    text_field.build_vocab(trainds, valds, vectors=vec, max_size=max_vocab_size)
    label_field.build_vocab(trainds)

    traindl, valdl, testdl = data.BucketIterator.splits(
        datasets=(trainds, valds, testds),
        batch_sizes=(batch_size_train, batch_size_validation, batch_size_test),
        sort_key=lambda x: x.sentence,
        repeat=False,
        device=device
    )
    print(
        "Number of Batches of train, validation and test data are respectively {}, {} and {}".format(
            len(traindl), len(valdl), len(testdl)
        )
    )
    print("Size of the vocabulary is {}".format(len(text_field.vocab.stoi)))
    return traindl, valdl, testdl, text_field, label_field

if __name__ == "__main__":
    traindl, valdl, testdl, text_field, label_field = prepare_data_loaders(
        sent_length=SENT_LENGTH,
        tokenizer_func=tokenizer_nltk,
        csv_field_names=CSV_FIELD_NAMES,
        csv_folder_path=CSV_FOLDER_PATH,
        train_file_name=TRAIN_FILE_NAME,
        validation_file_name=VALIDATION_FILE_NAME,
        test_file_name=TEST_FILE_NAME,
        glove_file_path=GLOVE_FILE_PATH,
        max_vocab_size=MAX_VOCAB_SIZE,
        batch_size_train=BATCH_SIZE_TRAIN,
        batch_size_validation=BATCH_SIZE_VALIDATION,
        batch_size_test=BATCH_SIZE_TEST,
        field_to_be_sorted_on=FIELD_TO_BE_SORTED_ON,
        device=DEVICE
    )
    SST2_DATASET_PARAMETERS["vocab_length"] = len(text_field.vocab.stoi)

    model = DCNN_SST(parameter_dict=SST2_DATASET_PARAMETERS)

    pretrained_embeddings = text_field.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    UNK_IDX = text_field.vocab.stoi[text_field.unk_token]
    PAD_IDX = text_field.vocab.stoi[text_field.pad_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    print_number_of_trainable_parameters(model)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model.to(DEVICE)
    criterion.to(DEVICE)

    print("Starting Training :::::::")
    base_dev_acc = 0.0
    for epoch in range(N_EPOCHS):
        is_best = False

        train_loss, train_acc = train(model, traindl, optimizer, criterion, device=DEVICE)
        valid_loss, valid_acc = evaluate(model, valdl, criterion, device=DEVICE)

        print(
            f'| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}% |')

        if base_dev_acc < valid_acc:
            is_best = True
            base_dev_acc = valid_acc

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': valid_loss,
            'best_dev_accuracy': valid_acc
        }, is_best, MODEL_PATH)

    print("Best validation set accuracy for the epochs is : {}".format(base_dev_acc))

    print("Prediction on Test :::::::")
    model = load_check_point(model, MODEL_PATH)
    test_loss, test_accuracy = evaluate(model, testdl, criterion, device=DEVICE)

    print(f"Test accuracy of the trained model is {test_accuracy * 100:.2f}%")
    print("\n\n")
    print("FINISH")
    print("############################################################################")

