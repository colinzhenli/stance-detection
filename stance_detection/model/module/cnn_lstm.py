import torch
import time
import torch.nn as nn
import numpy as np
import torchmetrics
import pytorch_lightning as pl


class CNN_LSTM(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_to_ix = {} # replaces words with an index (one-hot vector)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(64, hidden_dim, bidirectional=False)
        # The linear layer that maps from hidden state space to tag space
        self.conv1 = nn.Conv1d(100, 64, 5)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.functional.relu

        self.pool1 = torch.nn.MaxPool1d(4, 4)
        self.fc1 = nn.Linear(hidden_dim*1249, 500)
        self.fc2 = nn.Linear(500, tagset_size)
        self.sigmoid = nn.functional.sigmoid

    def forward(self, sentence):
        word_embeds = self.word_embeddings(sentence)
        word_embeds = word_embeds.permute(0, 2, 1)
        x = self.dropout(word_embeds)
        x = self.pool1(self.conv1(word_embeds))
        lstm_out, _ = self.lstm(x.permute(0, 2, 1))
        # x = self.sigmoid(self.fc1(lstm_out[:,-1,:])).clone()
        x = torch.flatten(lstm_out, start_dim=1)
        x = self.sigmoid(self.fc1(x)).clone()
        x = self.sigmoid(self.fc2(x)).clone()


        return x
