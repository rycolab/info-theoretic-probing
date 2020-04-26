import copy
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as t_utils
import torch.utils.data

from sklearn.decomposition import PCA

from .base import BaseModel
from util import util
from util import constants


class MLP(BaseModel):

    name = 'mlp'

    def __init__(self, task, embedding_size=768, n_classes=3, hidden_size=5,
                 nlayers=1, dropout=0.1, representation=None, n_words=None):
        super().__init__()

        # Save things to the model here
        self.dropout_p = dropout
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.n_classes = n_classes
        self.representation = representation
        self.n_words = n_words
        self.task = task

        if self.representation in ['onehot', 'random']:
            self.build_embeddings(n_words, embedding_size)

        self.mlp = self.build_mlp()
        self.out = nn.Linear(self.final_hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)

        self.criterion = nn.CrossEntropyLoss()

    def build_embeddings(self, n_words, embedding_size):
        if self.task == 'dep_label':
            self.embedding_size = int(embedding_size / 2) * 2
            self.embedding = nn.Embedding(n_words, int(embedding_size / 2))
        else:
            self.embedding = nn.Embedding(n_words, embedding_size)

        if self.representation == 'random':
            self.embedding.weight.requires_grad = False

    def build_mlp(self):
        src_size = self.embedding_size
        tgt_size = self.hidden_size
        mlp = []
        for layer in range(self.nlayers):
            mlp += [nn.Linear(src_size, tgt_size)]
            mlp += [nn.ReLU()]
            mlp += [nn.Dropout(self.dropout_p)]
            src_size, tgt_size = tgt_size, int(tgt_size / 2)
        self.final_hidden_size = src_size
        return nn.Sequential(*mlp)

    def forward(self, x):
        if self.representation in ['onehot', 'random']:
            x = self.get_embeddings(x)

        x_emb = self.dropout(x)
        x = self.mlp(x_emb)
        logits = self.out(x)
        return logits

    def get_embeddings(self, x):
        x_emb = self.embedding(x)
        if len(x.shape) > 1:
            x_emb = x_emb.reshape(x.shape[0], -1)

        return x_emb

    def train_batch(self, data, target, optimizer, criterion):
        optimizer.zero_grad()
        mlp_out = self(data)
        loss = self.criterion(mlp_out, target)
        loss.backward()
        optimizer.step()

        return loss.item() / math.log(2)

    def eval_batch(self, data, target):
        mlp_out = self(data)
        loss = self.criterion(mlp_out, target) / math.log(2)
        accuracy = (mlp_out.argmax(dim=-1) == target).float().detach().sum()
        loss = loss.item() * data.shape[0]

        return loss, accuracy

    def get_args(self):
        return {
            'nlayers': self.nlayers,
            'hidden_size': self.hidden_size,
            'embedding_size': self.embedding_size,
            'dropout': self.dropout_p,
            'n_classes': self.n_classes,
            'representation': self.representation,
            'n_words': self.n_words,
            'task': self.task,
        }
