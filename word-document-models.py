import sys

import csv
import itertools as it
import numpy as np
import sklearn.decomposition
import math
from matplotlib import pyplot as plt


class TfidfFeaturizer:
    def fit(self, matrix):
        self.idf = np.zeros((matrix.shape[0], matrix.shape[1]))
        docs_in_corpus = matrix.shape[1]

        docs_containing_word = []

        for i in range(matrix.shape[0]):
          total = 0
          for j in range(matrix.shape[1]):
            if matrix[i][j] >= 1:
              total += 1

          if total == 0:
            result = 0
          else:
            result = math.log(docs_in_corpus / total)

          self.idf[i][:] = result
        return self.idf


    def transform_tfidf(self, matrix):
        result = np.zeros((matrix.shape[0], matrix.shape[1]))
        for i in range(matrix.shape[0]):
          for j in range(matrix.shape[1]):
            result[i][j] = self.idf[i][j] * matrix[i][j]

        return result

td_matrix = bow_matrix.T
td_matrix_test = bow_matrix_test.T

featurizer = TfidfFeaturizer()
featurizer.fit(td_matrix)

featurizer1 = TfidfFeaturizer()
featurizer1.fit(td_matrix_test)

tfidf_matrix = featurizer.transform_tfidf(td_matrix)
tfidf_matrix_test = featurizer1.transform_tfidf(td_matrix_test)

from sklearn.linear_model import LogisticRegression

model_bow = LogisticRegression(max_iter=10000)
model_tfidf = LogisticRegression(max_iter=10000)

def train_and_eval(train_X, train_y, test_X, test_y, model):
    model.fit(train_X, train_y)
    model.predict(test_X)
    train_score = model.score(train_X, train_y)
    test_score = model.score(test_X, test_y)
    return (train_score, test_score, model)

print('Logistic regression with BoW')
print(train_and_eval(bow_matrix, train_labels, bow_matrix_test, test_labels, model_bow))

print('Logistic regression with TF-IDF features')
train_and_eval(tfidf_matrix.T, train_labels, tfidf_matrix_test.T, test_labels, model_tfidf)