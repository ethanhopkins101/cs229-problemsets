from typing import Any
import pandas as pd
import numpy as np
import string


class Vectorizer:
    def clean_str(self, s, c):
        table = str.maketrans("", "", c)
        result = s.translate(table)
        return result

    def clean_matrix(self, x):
        x = np.array(x)
        x = x.tolist()
        x = "".join(x)
        x = x.split(" ")
        for i, j in enumerate(x):
            x[i] = self.clean_str(j, self.undesirable)
        return x

    def clean_dict(self, x):
        x = np.array(x)
        x = x.tolist()
        x = "".join(x)
        x = x.split(" ")
        for i, j in enumerate(x):
            x[i] = self.clean_str(j, self.undesirable)
        x = list(set(x))
        return x

    def __init__(self):
        self.vocabulary = {}
        self.undesirable = string.punctuation + string.whitespace + string.digits

    def fit(self, x):
        x = self.clean_dict
        for i, j in enumerate(x):
            self.vocabulary[j] = i

    def get_feature_names_out(self):
        print(self.vocabulary.keys())

    def transform(self, x):
        x = np.array(x)
        temp = [0 for _ in range(len(self.vocabulary))]
        result = temp
        for i in range(x.shape[0]):
            x[i] = self.clean_matrix(x[i])
            for j, k in enumerate(x[i]):
                if k in self.vocabulary.keys():
                    temp[self.vocabulary[k]] += 1
            result = np.vstack(result, temp)
        return result[1:, :]
