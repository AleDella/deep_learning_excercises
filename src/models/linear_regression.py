import torch.nn as nn

class LinearRegressor(nn.Module):
    '''
    Module that implements a simple linear regression:
        y = X*w^T + b
        
    Args:
        input_dim (int): input dimension for the linear layer
        output_dim (int): output dimension for the linear layer
        softmax (bool): flag to use the softmax or not
    '''
    def __init__(self, input_dim, output_dim=1, softmax=False) -> None:
        super().__init__()
        if softmax:
            self.layer = nn.Sequential(
                nn.Linear(input_dim,output_dim),
                nn.Softmax(dim=1)
            )
        else:
            self.layer = nn.Linear(output_dim)
            
    
    def forward(self, x):
        x = x.float()
        return self.layer(x)