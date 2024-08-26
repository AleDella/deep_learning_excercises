import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    '''
    Inception module for GoogLeNet
    
    Args:
        oc1 (int): number of input channels in path 1
        oc2 ([int, int]): number of input channels in path 2
        oc3 ([int, int]): number of input channels in path 3
        oc4 (int): number of input channels in path 4
    '''
    def __init__(self, oc1, oc2, oc3, oc4) -> None:
        super(InceptionModule, self).__init__()
        self.p1 = nn.Sequential(
            nn.LazyConv2d(oc1, kernel_size=1),
            nn.ReLU()
        )
        self.p2 = nn.Sequential(
            nn.LazyConv2d(oc2[0], kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(oc2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.LazyConv2d(oc3[0], kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(oc3[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,padding=1,stride=1),
            nn.LazyConv2d(oc4, kernel_size=1),
            nn.ReLU()
        )
        
    def forward(self,x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        
        return torch.concatenate((p1,p2,p3,p4), dim=1)
    
class GoogLeNet(nn.Module):
    '''
    Class that implements GoogLeNet architecture
    
    Args:
        out_classes (int): output classes (default 10)
    '''
    
    def __init__(self, out_classes=10) -> None:
        super(GoogLeNet, self).__init__()
        self.layers = nn.Sequential(
            # First Block
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Second Block
            nn.LazyConv2d(64, kernel_size=1),
            nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Third Block
            InceptionModule(64, (96,128), (16,32), 32),
            InceptionModule(128, (128,192), (32,96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Fourth Block
            InceptionModule(192, (96,208), (16,48), 64),
            InceptionModule(160, (112,224), (24,64), 64),
            InceptionModule(128, (128,256), (24,64), 64),
            InceptionModule(112, (144,288), (32,64), 64),
            InceptionModule(256, (160,320), (32,128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Fifth Block
            InceptionModule(256, (160,320), (32,128), 128),
            InceptionModule(384, (192,384), (48,128), 128),
            nn.AvgPool2d(1),
            # Output Layer
            nn.Flatten(),
            nn.LazyLinear(out_classes)
        )
        
    def forward(self, x):
        return self.layers(x)