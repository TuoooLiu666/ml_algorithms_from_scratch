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

# initiate hyperparameters: learning rate
lr = 0.001

# create gradient descent function
def gradient_descent(X, y, w=w, b=b, lr=lr, n_iter=500):
    n_samples = X.shape[0]
    
    for i in range(n_iter):    
        
        # prediction
        y_predicted = np.dot(X, w) + b
        #  loss
        loss = np.mean((y - y_predicted)**2)
        # calcualte derivatives
        dw = -2 * np.dot(X.T, (y - y_predicted)) * (1/n_samples)
        db = -2 * np.sum(y - y_predicted) * (1/n_samples)
        
        # UPDATE PARAMETERS by dradient descent rule
        w -= lr * dw 
        b -= lr * db 
        
        print(f"Iteration {i}: loss = {loss:.4f}, weights = {w[0][0]:.4f}, bias = {b:.4f}")    
    
gradient_descent(X=X, y=y, lr=0.01, n_iter=500)
