# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
import random

word_freq = Counter()
word_ix = {}

tag_to_idx = {
    "``": 0,
    "VBZ": 1,
    "FW": 2,
    "JJ": 3,
    "#": 4,
    "POS": 5,
    "-LRB-": 6,
    "VB": 7,
    "MD": 8,
    "PRP": 9,
    "EX": 10,
    ".": 11,
    "RP": 12,
    "IN": 13,
    "WP$": 14,
    "CD": 15,
    "DT": 16,
    "CC": 17,
    "JJR": 18,
    "RBS": 19,
    "JJS": 20,
    "NN": 21,
    "NNS": 22,
    "NNPS": 23,
    "RB": 24,
    "VBP": 25,
    "WDT": 26,
    "PDT": 27,
    "VBN": 28,
    "VBD": 29,
    "WP": 30,
    ",": 31,
    "TO": 32,
    ":": 33,
    "-RRB-": 34,
    "WRB": 35,
    "RBR": 36,
    "PRP$": 37,
    "SYM": 38,
    "$": 39,
    "''": 40,
    "VBG": 41,
    "UH": 42,
    "NNP": 43,
    "LS": 44
}
CHAR_SIZE = 128  # Ascii
MAX_SENTENCE_LEN = 150
MAX_WORD_LEN = 45
CHAR_EMBEDDING_DIM = 256
WORD_EMBEDDING_DIM = 300
HIDDEN_DIM = 0
NUM_OF_FILTERS = 0
PADDING = 1
TAGSET_SIZE = 45
WINDOW_SIZE = 3
LSTM_LAYERS = 1
DROPOUT_PROB = 0
BATCH_SIZE = 128
NUM_EPOCHS = 30


class CNNBiLSTMTagger(nn.Module):

    def __init__(self, vocab_size, tagset_size, char_embedding_dim, word_embedding_dim, hidden_dim, dropout,
                 num_of_filters, window_size, padding, num_layers):
        super(CNNBiLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_embeddings = nn.Embedding(CHAR_SIZE, char_embedding_dim, padding_idx=0)
        # padding = 1 to preserve input shape. (#batch, #chars, #char_features)
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=CHAR_SIZE, out_channels=num_of_filters,
                                             kernel_size=window_size, padding=padding), nn.Sigmoid(),
                                   nn.MaxPool1d(MAX_WORD_LEN))
        self.word_embeddings = nn.Embedding(vocab_size + 2, word_embedding_dim, padding_idx=vocab_size + 1)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.biLSTM = nn.LSTM(num_of_filters + word_embedding_dim, hidden_dim, num_layers=num_layers, bias=True,
                              batch_first=True, dropout=dropout, bidirectional=True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2*hidden_dim, tagset_size)

    def forward(self, sentence):
        # Add your code here for character CNN and BiLSTM
        word_tensor, char_tensor = sentence
        word_embed = self.word_embeddings(word_tensor)
        word_embed = torch.transpose(word_embed, 1, 2)
        char_embed = self.char_embeddings(char_tensor)
        char_embed = torch.transpose(char_embed, 1, 2)
        char_conv_output = self.conv1(char_embed)
        #Shape (#batches, #filters + #word features, # words)
        lstm_input = torch.cat([word_embed, char_conv_output])
        lstm_input = torch.transpose(lstm_input, 1, 2)
        lstm_output, _ = self.biLSTM(lstm_input)
        tag_space = self.hidden2tag(lstm_output)
        tag_scores = F.log_softmax(tag_space, dim=2)
        return tag_scores


def init_random_seeds():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True


def get_char_embedding_ix(char):
    return ord(char) % 128


def process_line(line):
    global word_freq
    words, tags, word_counter = [], [], Counter()
    for ele in line.split():
        temp = ele.split("/")
        tag = temp[-1]
        word = "/".join(temp[0:-1])
        word_counter[word] += 1
        words.append(word)
        tags.append(tag)
    word_freq += word_counter
    return words, tags


def encode_sent_as_char_tensor(sentence):
    res_tensor = [0 for _ in range(MAX_SENTENCE_LEN * MAX_WORD_LEN)]
    starting_index = 0
    for word in sentence:
        j = 0
        for char in word:
            res_tensor[starting_index + j] = get_char_embedding_ix(char)
            j += 1
        starting_index += MAX_WORD_LEN
    return torch.tensor(res_tensor, dtype=torch.long)


def encode_input_vector_as_tensors(training_words):
    last_count = 0
    tensors = []
    global word_ix
    for sentence in training_words:
        input_tensor = [0 for _ in range(len(sentence))]
        for index, word in enumerate(sentence):
            if word_freq[word] > 1:
                if word not in word_ix:
                    last_count += 1
                    word_ix[word] = last_count
                input_tensor[index] = word_ix[word]
        tensors.append((torch.tensor(input_tensor, dtype=torch.long), encode_sent_as_char_tensor(sentence)))
    return tensors, last_count


def pad_input_vectors(input_vectors, vocab_size):
    for index, (word_vector, char_vector) in enumerate(input_vectors):
        pad_len = MAX_SENTENCE_LEN - word_vector.shape[0]
        input_vectors[index] = (F.pad(word_vector, [0, pad_len], vocab_size + 1), char_vector)
    return input_vectors


def pad_output_vectors(output_vectors):
    for index, pos_tag_vector in enumerate(output_vectors):
        pad_len = MAX_SENTENCE_LEN - pos_tag_vector.shape[0]
        output_vectors[index] = F.pad(pos_tag_vector, [0, pad_len])
    return output_vectors


def encode_output_tags_as_tensor(tags):
    return [torch.tensor([tag_to_idx[t] for t in tag_list], dtype=torch.long) for tag_list in tags]


def batch_input(input_vectors, output_vectors):
    num_samples = len(input_vectors)
    permutation = torch.randperm(num_samples)
    for i in range(0, num_samples, BATCH_SIZE):
        indices = permutation[i : i + BATCH_SIZE]
        yield [input_vectors[x] for x in indices], [output_vectors[y] for y in indices]


def train_model(train_file, model_file):
    init_random_seeds()
    with open(train_file, "r") as f:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lines = f.readlines()
        input = [process_line(line) for line in lines]
        # vocab_size exclude unk token.
        input_vectors, vocab_size = encode_input_vector_as_tensors([t[0] for t in input])
        input_vectors = pad_input_vectors(input_vectors, vocab_size)
        output_vectors = encode_output_tags_as_tensor([t[1] for t in input])

        model = CNNBiLSTMTagger(vocab_size, TAGSET_SIZE, CHAR_EMBEDDING_DIM, WORD_EMBEDDING_DIM, HIDDEN_DIM,
                                DROPOUT_PROB, NUM_OF_FILTERS, WINDOW_SIZE, PADDING, LSTM_LAYERS)
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)

        for epoch in range(NUM_EPOCHS):
            for batch_x, batch_y in batch_input(input_vectors, output_vectors):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 2. Run our forward pass.
                tag_scores = model(batch_x)
                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                batch_size, num_words, num_classes = tag_scores.shape
                loss = loss_function(tag_scores.reshape(batch_size * num_words, num_classes),
                              batch_y.reshape(BATCH_SIZE * num_words))
                loss.backward()
                optimizer.step()

    torch.save((word_ix, model.state_dict()), model_file)
        # input_vectors

    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
