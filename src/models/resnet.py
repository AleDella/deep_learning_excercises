import torch
import torch.nn as nn

class ResidualModule(nn.Module):
    '''
    Class that implements a residual module
    
    Args:
        channel_dim (int): inner channel dimension
        stride (int): stride applied to the first convolution and on the one on the skip connection
        use1x1 (bool): flag to use a 1x1 convolution on the residual connection or not
    '''
    def __init__(self, channel_dim, stride=1, use1x1=False) -> None:
        super(ResidualModule, self).__init__()
        self.convo1 = nn.LazyConv2d(channel_dim, kernel_size=3, padding=1, stride=stride)
        self.convo2 = nn.LazyConv2d(channel_dim, kernel_size=3, padding=1)
        # Use a convolution 1x1 in the skip connection before merging
        if use1x1:
            self.convo3 = nn.LazyConv2d(channel_dim, kernel_size=1, stride=stride)
        else:
            self.convo3 = None
        
        self.bn1 = nn.BatchNorm2d(channel_dim)
        self.bn2 = nn.BatchNorm2d(channel_dim)
        
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        out = self.bn2(self.convo2(self.relu(self.bn1(self.convo1(x)))))
        if self.convo3:
            x = self.convo3(x)
        
        return self.relu(out+x)
    
def residual_block(num_channels, num_residuals, first_block=False):
    '''
    Function that creates a residual block of a given depth
    '''
    block = nn.Sequential()
    
    for i in range(num_residuals):
        if i==0 and not first_block:
            block.append(ResidualModule(num_channels, stride=2, use1x1=True))
        else:
            block.append(ResidualModule(num_channels))
    
    return block

class ResNet(nn.Module):
    '''
    Class implementing a ResNet
    
    Args:
        output_dim (int): Number of classes
    '''
    def __init__(self, output_dim=10) -> None:
        super(ResNet, self).__init__()
        self.layers = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            residual_block(64,2,True),
            residual_block(128,2),
            residual_block(256,2),
            residual_block(512,2),
            nn.AvgPool2d(3),
            nn.Flatten(),
            nn.LazyLinear(output_dim)
        )
        
    def forward(self,x):
        return self.layers(x)