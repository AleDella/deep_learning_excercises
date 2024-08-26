import torch
from torch.nn import MSELoss

def accuracy(pred, truth):
    '''
    Function to compute the accuracy
    
    Args:
        pred (torch.Tensor): prediction of the network
        truth (torch.Tensor): label of the network
    Returns:
        accuracy value
    '''
    if len(pred.shape)>1 and pred.shape[1]>1:
        pred = pred.argmax(axis=1)
    cmp = pred.type(truth.dtype) == truth
    return float(cmp.type(truth.dtype).sum())/len(cmp)

def log_rmse(pred, labels):
    loss = MSELoss()
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = torch.clip(pred, 1, float('inf'))
    return torch.sqrt(2 * loss(torch.log(clipped_preds), torch.log(labels)).mean())