import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

# load in binary dataset
bc = datasets.load_breast_cancer()
X,y = bc.data,bc.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)

def accuracy(y_test, y_predicted):
    acc = np.sum(y_test == y_predicted) / len(y_test)
    return acc

# initialize model
regressor = LogisticRegression(lr=0.001, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print("Logistic Regression Accuracy: ", accuracy(y_test, predictions))