import pandas as pd
import numpy as np


class LSBernoulliNB:
    def __init__(self):
        self.pxy1 = self.pxy0 = self.py0 = self.py1 = 0

    def fit(self, x, y):
        x_spam = x[y[:, 0] == 1]
        x_ham = x[y[:, 0] == 0]
        self.pxy1 = self.pxy0 = [0 for _ in range(x.shape[1])]

        for i in range(x.shape[1]):
            self.pxy1[i] = (np.sum(x_spam[:, i]) + 1) / (x_spam.shape[0] + 2)
            self.pxy0[i] = (np.sum(x_ham[:, i]) + 1) / (x_ham.shape[0] + 2)
        self.py1 = x_spam.shape[0] / x.shape[0]
        self.py0 = 1 - self.py1

    def predict(self, x):
        temp_yes = []
        temp_no = []
        for i, j in enumerate(x):
            if j == 1:
                temp_yes.append(self.pxy1[i])
                temp_no.append(self.pxy0[i])
        yes = self.py1 * np.prod(temp_yes)
        no = self.py0 * np.prod(temp_no)
        return 1 if yes > no else 0
