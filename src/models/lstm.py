import torch
import torch.nn as nn


class LSTM(nn.Module):
    '''
    Function that implements a LSTM network
    
    Args:
        vocab_length (int): length of the vocabulary
        hidden_dim (int): dimension for the hidden layers
        device (str): torch device
    '''
    def __init__(self, vocab_length, hidden_dim=256, layers=1, device='cpu') -> None:
        super(LSTM, self).__init__()
        self.vocab_length = vocab_length
        self.device = device
        self.layers = nn.LSTM(vocab_length, hidden_size=hidden_dim, num_layers=layers)
        self.to(device)
        
    def forward(self, x, H, C):
        H = H.to(self.device)
        C = C.to(self.device)
        x = nn.functional.one_hot(x, self.vocab_length)
        x = x.float()
        x = x.to(self.device)
        return self.layers(x, (H, C))