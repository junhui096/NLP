# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

idx_to_tag = {
    0: "``",
    1: "VBZ",
    2: "FW",
    3: "JJ",
    4: "#",
    5: "POS",
    6: "-LRB-",
    7: "VB",
    8: "MD",
    9: "PRP",
    10: "EX",
    11: ".",
    12: "RP",
    13: "IN",
    14: "WP$",
    15: "CD",
    16: "DT",
    17: "CC",
    18: "JJR",
    19: "RBS",
    20: "JJS",
    21: "NN",
    22: "NNS",
    23: "NNPS",
    24: "RB",
    25: "VBP",
    26: "WDT",
    27: "PDT",
    28: "VBN",
    29: "VBD",
    30: "WP",
    31: ",",
    32: "TO",
    33: ":",
    34: "-RRB-",
    35: "WRB",
    36: "RBR",
    37: "PRP$",
    38: "SYM",
    39: "$",
    40: "''",
    41: "VBG",
    42: "UH",
    43: "NNP",
    44: "LS"
}
CHAR_SIZE = 128  # Ascii
MAX_SENTENCE_LEN = 150
MAX_WORD_LEN = 45
CHAR_EMBEDDING_DIM = 256
WORD_EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_OF_FILTERS = 7
PADDING = 1
TAGSET_SIZE = 45
WINDOW_SIZE = 3
LSTM_LAYERS = 3
DROPOUT_PROB = 0.5
BATCH_SIZE = 128
NUM_EPOCHS = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNNBiLSTMTagger(nn.Module):

    def __init__(self, vocab_size, tagset_size, char_embedding_dim, word_embedding_dim, hidden_dim, dropout,
                 num_of_filters, window_size, padding, num_layers, conv_dropout_prob=0.2):
        super(CNNBiLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.char_embeddings = nn.Embedding(CHAR_SIZE, char_embedding_dim, padding_idx=0)
        # padding = 1 to preserve input shape. (#batch, #chars, #char_features)
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=CHAR_EMBEDDING_DIM, out_channels=num_of_filters,
                                             kernel_size=window_size, padding=padding), nn.Sigmoid(),
                                   nn.MaxPool1d(MAX_WORD_LEN), nn.Dropout(conv_dropout_prob))
        self.word_embeddings = nn.Embedding(vocab_size + 2, word_embedding_dim, padding_idx=vocab_size + 1)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.biLSTM = nn.LSTM(num_of_filters + word_embedding_dim, hidden_dim, num_layers=num_layers, bias=True,
                              batch_first=True, dropout=dropout, bidirectional=True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2 * hidden_dim, tagset_size)

        self.fc_dropout = nn.Dropout(dropout)

    def forward(self, sentence):
        # Add your code here for character CNN and BiLSTM
        word_tensor, char_tensor = sentence
        word_embed = self.word_embeddings(word_tensor)
        word_embed = torch.transpose(word_embed, 1, 2)
        char_embed = self.char_embeddings(char_tensor)
        char_embed = torch.transpose(char_embed, 1, 2)
        char_conv_output = self.conv1(char_embed)
        # Shape (#batches, #filters + #word features, # words)model.load_state_dict(model_state_dict)
        lstm_input = torch.cat([word_embed, char_conv_output], dim=1)
        lstm_input = torch.transpose(lstm_input, 1, 2)
        lstm_output, _ = self.biLSTM(lstm_input)
        tag_space = self.hidden2tag(lstm_output)
        tag_scores = F.log_softmax(tag_space, dim=2)
        return tag_scores


def get_char_embedding_ix(char):
    return ord(char) % 128


def encode_sent_as_char_tensor(sentence):
    res_tensor = [0 for _ in range(MAX_SENTENCE_LEN * MAX_WORD_LEN)]
    starting_index = 0
    for word in sentence:
        j = 0
        for char in word:
            res_tensor[starting_index + j] = get_char_embedding_ix(char)
            j += 1
        starting_index += MAX_WORD_LEN
    return torch.tensor(res_tensor, dtype=torch.long, device=device)


def encode_sent_as_word_tensor(sentence, vocab, vocab_size):
    res = [vocab_size + 1 for _ in range(MAX_SENTENCE_LEN)]
    for index, word in enumerate(sentence):
        if word in vocab:
            res[index] = vocab[word]
        else:
            res[index] = 0
    return torch.tensor(res, dtype=torch.long, device=device)


def process_sentence(vocab, vocab_size, sentence):
    sentence = sentence.split()
    char_tensor = encode_sent_as_char_tensor(sentence)
    word_tensor = encode_sent_as_word_tensor(sentence, vocab, vocab_size)
    return word_tensor, char_tensor, sentence


def get_next_tag(indices):
    for k in indices:
        yield idx_to_tag[k.item()]


def get_output(sentence, model):
    word_tensor, char_tensor, words_list = sentence
    with torch.no_grad():
        tag_scores = model((torch.stack([word_tensor]), torch.stack([char_tensor]))).squeeze()
        _, indices = torch.topk(tag_scores, 1)
        return " ".join(['/'.join(t) for t in zip(words_list, get_next_tag(indices))]) + "\n"


def tag_sentence(test_file, model_file, out_file):
    vocab, model_state_dict = torch.load(model_file)
    vocab_size = len(vocab)
    model = CNNBiLSTMTagger(vocab_size, TAGSET_SIZE, CHAR_EMBEDDING_DIM, WORD_EMBEDDING_DIM, HIDDEN_DIM,
                            DROPOUT_PROB, NUM_OF_FILTERS, WINDOW_SIZE, PADDING, LSTM_LAYERS)
    model.load_state_dict(model_state_dict)
    model.to(device=device)
    model.eval()
    result = []
    with open(test_file, "r") as f:
        sentence = process_sentence(vocab, vocab_size, f.readline())
        while sentence:
            result.append(get_output(sentence, model))
            sentence = process_sentence(vocab, vocab_size, f.readline())
    with open(out_file, "w") as g:
        g.writelines(result)

    # write your code here. You can add functions as well.
    # use torch library to load model_file
    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
