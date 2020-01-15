import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SentimentLSTM(nn.Module):

    def __init__(self, n_vocab, n_embed, n_hidden, n_output, n_layers, drop_p=0.5):
        super().__init__()
        # params: "n_" means dimension
        self.n_vocab = n_vocab  # number of unique words in vocabulary
        self.n_layers = n_layers  # number of LSTM layers
        self.n_hidden = n_hidden  # number of hidden nodes in LSTM
        self.embedding = nn.Embedding(n_vocab, n_embed)
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, batch_first=True, dropout=drop_p)
        self.dropout = nn.Dropout(drop_p)
        self.fc = nn.Linear(n_hidden, n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_words, hidden):
        # INPUT   :  (batch_size, seq_length)
        batch_size = input_words.size(0)
        embedded_words = self.embedding(input_words)  # (batch_size, seq_length, n_embed)
        lstm_out, h = self.lstm(embedded_words)  # (batch_size, seq_length, n_hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.n_hidden)  # (batch_size*seq_length, n_hidden)

        fc_out = self.fc(lstm_out)  # (batch_size*seq_length, n_output)
        sigmoid_out = self.sigmoid(fc_out)  # (batch_size*seq_length, n_output)
        sigmoid_out = sigmoid_out.view(batch_size, -1)  # (batch_size, seq_length*n_output)

        # extract the output of ONLY the LAST output of the LAST element of the sequence
        sigmoid_last = sigmoid_out[:, -1]  # (batch_size, 1)

        return sigmoid_last, h

    def init_hidden(self, batch_size):  # initialize hidden weights (h,c) to 0

        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_(),
             weights.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return h


class BiLSTM(nn.Module):
    pass


if __name__ == "__main__":
    n_vocab = 46139
    n_embed = 400
    n_hidden = 512
    n_output = 1  # 1 ("positive") or 0 ("negative")
    n_layers = 2
    net = SentimentLSTM(n_vocab, n_embed, n_hidden, n_output, n_layers)
    print(net)
