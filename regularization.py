import numpy as np

def standard_scaler(X):
    means = X.mean(0)
    stds = X.std(0)
    return (X - means)/stds

def sign(x, first_element_zero = False):
    signs = (-1)**(x < 0)
    if first_element_zero:
        signs[0] = 0
    return signs

class RegularizedRegression:
    def _record_info(self, X, y, lam: int, intercept, standardize):
        
        # standardize
        if standardize == True:
            X = standard_scaler(X)

        # add intercept
        if intercept == False:
            ones = np.ones(len(X)).reshape(len(X), 1) # column of ones
            X = np.concatenate((ones, X), axis=1) # add column of ones to X
            
        # store info
        self.X = np.array(X)
        self.y = np.array(y)
        self.N, self.D = self.X.shape
        self.lam = lam
        
    def fit_ridge(self, X, y, lam=0, intercept=False, standardize=True):
        # record data and dimensions
        self._record_info(X, y, lam, intercept, standardize)
        
        # estimate parameters
        XtX = np.dot(self.X.T, self.X)
        I_prime = np.eye(self.D) 
        I_prime[0,0] = 0 # don't regularize intercept
        XtX_plus_lam_inverse = np.linalg.inv(XtX + self.lam*I_prime)
        Xty = np.dot(self.X.T, self.y)
        self.beta_hats = np.dot(XtX_plus_lam_inverse, Xty)
        
        # fitted values
        self.y_hat = np.dot(self.X, self.beta_hats)
        
    def fit_lasso(self, X, y, lam=0, n_iters=2000, 
                  lr=0.001, intercept=False, standardize=True):
        # record data and dimensions
        self._record_info(X, y, lam, intercept, standardize)
        
        # estimate parameters
        beta_hats = np.random.randn(self.D)
        for i in range(n_iters):
            dL_dbeta = -self.X.T @ (self.y - self.X @ beta_hats) + self.lam * sign(beta_hats, True)
            beta_hats -= lr * dL_dbeta
        self.beta_hats = beta_hats
        
        # fitted values
        self.y_hat = np.dot(self.X, self.beta_hats)
        
