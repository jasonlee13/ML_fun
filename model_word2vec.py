# Word2Vec Model Implementation

import csv
import itertools as it
import numpy as np
import sklearn.decomposition
import math
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torch_data


class Word2VecModel(nn.Module):
    def __init__(self, corpus_size, em_dimensions, padding_idx):
        super().__init__()

        self.embeddings = nn.Embedding(num_embeddings = corpus_size+1, embedding_dim = em_dimensions, padding_idx = corpus_size)
        self.linear = nn.Linear(in_features=em_dimensions, out_features=corpus_size)

    def forward(self, context):
        x = self.embeddings(context)
        x = self.linear(x.mean(axis=1))
        x = F.log_softmax(x, dim=1)

        return x

def word2vec_learn(corpus, window_size, rep_size, n_epochs, n_batch):

    tokenizer = Tokenizer()
    tokenizer.fit(corpus)
    tokenized_corpus = tokenizer.tokenize(corpus)

    ngrams = get_ngrams(tokenized_corpus, window_size, pad_idx=2006)

    device = torch.device('cuda') 
    model = Word2VecModel(tokenizer.vocab_size, rep_size).to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)

    loader = torch_data.DataLoader(ngrams, batch_size=n_batch, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()  

    losses = [] 
    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0
        for context, label in loader:

            model.zero_grad()
            preds = model(context.cuda()).cuda()
            label = label.cuda()

            loss = loss_fn(preds, label)
            loss.backward()

            opt.step()
            opt.zero_grad()

            epoch_loss += loss.item()
            
        losses.append(epoch_loss)

    embedding_matrix = []
    for layer in model.embeddings.parameters():
      embedding_matrix.append(layer.data.cpu().numpy())

    return embedding_matrix