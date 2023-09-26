# Latent Semantic Analysis (LSA) Representation

import sys
import csv
import itertools as it
import numpy as np
import sklearn.decomposition
import math
from matplotlib import pyplot as plt

from sklearn.decomposition import TruncatedSVD

import sklearn.linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def learn_lsa(matrix, rep_size):
    svd = TruncatedSVD(n_components=rep_size)
    result = svd.fit_transform(matrix)
    return result

def featurize(x):
    return np.dot(x, matrix) / np.sqrt((np.dot(x, matrix) ** 2).sum(axis=1, keepdims=True))

def train(featurizer, x, y):
    x = featurizer(x)
    model = sklearn.linear_model.LogisticRegression(penalty='none', max_iter=1000)
    model.fit(x, y)
    return model

def evaluate(model, featurizer, x, y):
    x = featurizer(x)
    y_pred = model.predict(x)
    return np.mean(y_pred == y)

def testing(name, featurizer, n_train):
    x_train = vectorizer.transform(data[:n_train])
    y_train = data_labels[:n_train]
    x_test = vectorizer.transform(test_data)
    y_test = test_data

    model = train(featurizer, x_train, y_train)
    results = eval(model, featurizer, x_test, y_test)

    return results