import numpy as np

def bootstrap(data, num_samples, statistic=np.mean):
    """
    Perform bootstrapping on the given data.

    Args:
    - data (list or numpy array): The input data.
    - num_samples (int): The number of bootstrap samples.
    - statistic (function, optional): The statistic to compute. Defaults to np.mean.

    Returns:
    - numpy array: An array of bootstrap sample statistics.
    """
    # Resample with replacement
    resamples = np.random.choice(data, size=(num_samples, len(data)), replace=True)
    
    # Compute the statistic for each resample
    bootstrap_samples = np.apply_along_axis(statistic, axis=1, arr=resamples)
    return bootstrap_samples

# Example usage:
data = np.array(np.random.randn(1000))
num_samples = 100
bootstrap_samples = bootstrap(data, num_samples)
print("Bootstrap samples:", bootstrap_samples)