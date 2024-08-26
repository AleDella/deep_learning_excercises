import torch.nn as nn
import torch

class MLP(nn.Module):
    '''
    Class to implement a simple multi-layer perceptron
    
    Args:
        input_dim (int): input dimension
        output_dim (int): output dimension
    '''
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256,output_dim)
        )
    
    def forward(self,x):
        return self.layers(torch.flatten(x, start_dim=1))