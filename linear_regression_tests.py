import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets


X,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:,0], y, color='b', marker="o", s=30)
# plt.show()

from simple_linear_regression import LinearRegression

regressor = LinearRegression(lr=0.1)
fit = regressor.fit(X_train, y_train)
y_predicted = regressor.predict(X_test)

def mse(y_test, y_predicted) -> float:
    return np.mean((y_test-y_predicted)**2)

def r2(y_test, y_predicted) -> float:
    ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
    ss_residual = np.sum((y_test - y_predicted) ** 2)
    return 1 - (ss_residual / ss_total)

def adj_r2(y_test, y_predicted, n: int, p: int) -> float:
    ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
    ss_residual = np.sum((y_test - y_predicted) ** 2)
    r2_score = 1 - (ss_residual / ss_total)
    return 1 - ((1 - r2_score) * (n - 1) / (n - 1 - p))

mse_test = mse(y_test=y_test, y_predicted=y_predicted)
r2_test = r2(y_test=y_test, y_predicted=y_predicted)
r2_test_adj = adj_r2(y_test=y_test, y_predicted=y_predicted, n=len(y_test), p=1)
print(mse_test)
print(r2_test)
print(r2_test_adj)

# plot the line
y_pred_line = regressor.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()
