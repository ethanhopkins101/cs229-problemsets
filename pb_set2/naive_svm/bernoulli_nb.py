import pandas as pd
import numpy as np


class BernoulliNB:
    def __init__(self):
        self.py0 = self.py1 = 0
        self.pxy1 = self.pxy0 =[]
    def fit(self, x_train, y):
        # Transforming x_train to a matrix of one's '1' and zeroes '0'
        # In order to perform bernoulli opperation
        x = x_train
        for i in range(x_train.shape[0]):
            for j, k in enumerate(x_train):
                if k != 0:
                    x[i, j] = 1
        # splitting our data into spam and ham email , to make easier to train
        x_spam = x[y[:, 0] == 1]
        x_ham = x[y[:, 0] == 0]
        # Training our classifier with all given training examples 'm'
        for i in range(x.shape[1]):
            self.pxy1.append(np.sum(x_spam[:, i]) / x_spam.shape[0])
            self.pxy0.append(np.sum(x_ham[:, i]) / x_ham.shape[0])
        self.py1 = x_spam.shape[0] / x.shape[0]
        self.py0 = 1 - self.py1

    def predict(self, x):
        temp_yes = temp_no = []
        y_pred = []
        # Retreiving probabilities based on words appearences in our message
        for i in range(x.shape[0]):
            # For each training example
            for j, k in enumerate(x):
                # For each word
                if k != 0:
                    temp_yes.append(self.pxy1[j])
                    temp_no.append(self.pxy0[j])
            yes = self.py1 * np.prod(temp_yes)
            no = self.py0 * np.prod(temp_no)
            # Predict y=1 or y=0 ie spam/ham with respect to probabilties
            y_pred.append(1 if yes > no else 0)
        return y_pred
