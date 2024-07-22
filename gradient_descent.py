# gradient descent for linear regression
# yhat = wx+b
# loss = (yhat-y)**2

import numpy as np

# initiate sample data
X  = np.random.randn(10,1)
y = 4 + 3 * X + np.random.rand()

# initiate parameters
w = 0.0
b = 0.0

# initiate hyperparameters
lr = 0.001

# create gradient descent function
def gradient_descent(X, y, w, b, lr):
    n_samples, n_featrue = X.shape
    for xi, yi in zip(X, y):
        # dw = 2(yhat-(wx+b))(-x)
        dw = 2 * X.T.dot(X.dot(w) + b - y)
        # db = 2(yhat-(wx+b))(-1)
        db = 2 * np.sum(X.dot(w) + b - y) 
        
    w -= lr * dw * (1/n_samples)
    b -= lr * db * (1/n_samples)
    
    return w,b

