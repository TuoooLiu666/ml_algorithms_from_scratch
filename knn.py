import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class knn:
    
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.Y_train = y
    
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    
    def _predict(self, x):
        # Compute the distance between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # get k nearest samples, labels by sorting
        k_idx = np.argsort(distances)[:self.k]
        k_labels = [self.Y_train[i] for i in k_idx]
        
        # majority vote
        votes = Counter(k_labels).most_common(1)[0][0]
        return votes
    
    
    
    