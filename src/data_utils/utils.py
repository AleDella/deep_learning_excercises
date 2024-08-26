import random
import torch
import numpy as np


def synthetic_data(w, b, num_examples):
    """
    Generate y = Xw + b + noise.
    Used for toy-dataset generation.
    """
    X = torch.randn((num_examples, len(w)))
    y = X*w + b
    y += torch.randn(y.shape)
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    '''
    Function to iterate through the dataset in batches manually
    '''
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]