from statistics import mean
import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        m=y.shape[0]
        feta=np.sum(y)/y.shape[0]
        alfa=np.argwhere(y==0)
        beta=np.argwhere(y==1)
        mu0=x[alfa]/len(alfa)
        mu1=x[beta]/len(beta)
        mean_vector=np.array([[1 for _ in range(y.shape[0])]])
        for i in range(y.shape[0]):
            if y[i]==1:
                mean_vector[i]=mu1
            else:
                mean_vector[i]=mu0
        sigma=(1/m)*np.dot(x-mean_vector,(x-mean_vector).T)
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE
