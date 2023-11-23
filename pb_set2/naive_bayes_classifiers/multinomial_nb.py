import numpy as np


class MultinomialNB:
    def __init__(self):
        self.py1 = self.py0 = 0
        self.pxy1 = self.pxy0 = []

    def fit(self, x, y):
        # splitting our data into spam and ham email , to make easier to train
        x_spam = x[y[:, 0] == 1]
        x_ham = x[y[:, 0] == 0]
        self.pxy1 = self.pxy0 = x.shape[1]
        self.py1 = np.sum(y) / y.shape[0]
        self.py0 = 1 - self.py1
        # Training our classifier with all given training examples 'm'
        for i in range(x.shape[1]):
            self.pxy1.append(
                (np.sum(x_spam[:, i]) + 1) / ((x_spam.shape[0]) + x.shape[1])
            )
            self.pxy0.append(
                (np.sum(x_ham[:, i] + 1) / ((x_ham.shape[0]) + x.shape[1]))
            )

    def predict(self, x):
        y_pred = []
        temp_yes = temp_no = []
        # Retreiving probabilities based on words appearences in our message
        for i in range(x.shape[0]):
            # for each message
            for j, k in enumerate(x[i, :]):
                # for each word
                if k != 0:
                    temp_yes.append(self.pxy1[j])
                    temp_no.append(self.pxy0[j])
            yes = np.prod(temp_yes) * self.py1
            no = np.prod(temp_no) * self.py0
            # Predict y=1 or y=0 ie spam/ham with respect to probabilties
            y_pred.append(1 if yes > no else 0)
        return y_pred
