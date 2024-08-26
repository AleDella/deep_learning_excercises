import torch

def sgd(parameters, lr):
    """Minibatch stochastic gradient descent."""
    return torch.optim.SGD(parameters, lr=lr)
    