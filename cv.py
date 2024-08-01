import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# train-test split
from random import seed, randrange

def train_test_split(dataset, test_size=0.2, random_state=None):
    train = list()
    dataset_copy = list(dataset)
    train_size = round(len(dataset) * (1.0 - test_size))
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy

# k-fold split
def k_fold_split(dataset, k=5, random_state=None):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = len(dataset) // k 
    for i in range(k):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
    return dataset_split


def nested_cross_validation(X, y, outer_folds=5, inner_folds=5,
                            algorithm=KNeighborsClassifier, 
                            param_grid=None):
    if param_grid is None:
        param_grid = {'n_neighbors': [3, 5, 7, 9]}
    
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)

    outer_results = []   
    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        best_score = -np.inf
        best_params = None
        
        for params in ParameterGrid(param_grid):
            inner_scores = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
                X_inner_train, X_val = X_train[inner_train_idx], X_train[inner_val_idx]
                y_inner_train, y_val = y_train[inner_train_idx], y_train[inner_val_idx]
                
                model = algorithm(**params)
                model.fit(X_inner_train, y_inner_train)
                predictions = model.predict(X_val)
                score = accuracy_score(y_val, predictions)
                inner_scores.append(score)
            
            mean_inner_score = np.mean(inner_scores)
            
            if mean_inner_score > best_score:
                best_score = mean_inner_score
                best_params = params
        
        best_model = algorithm(**best_params)
        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)
        outer_score = accuracy_score(y_test, predictions)
        outer_results.append(outer_score)
    
    return outer_results, np.mean(outer_results), np.std(outer_results)



if __name__ == '__main__':
    from random import randrange
    from random import seed
    
    seed(2024)
    dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    train, test = train_test_split(dataset, test_size=0.4)
    print(train)
    print(test)
    
    folds = k_fold_split(dataset, k=4)
    print(folds)