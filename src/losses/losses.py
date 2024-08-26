import torch

def squared_loss(y_hat, y):
    """Squared loss function."""
    l = torch.nn.MSELoss()
    return l(y,y_hat)