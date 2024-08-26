import torch
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader,random_split
from torchvision.transforms import v2
from torch.optim.adam import Adam
from torch.nn import CrossEntropyLoss

from models import GoogLeNet
from training import train_network, test_network
from metrics import accuracy

if __name__ == '__main__':
    # Hyperparams
    dataset_root = "datasets\\FashionMNIST"
    validation_ratio = 0.3
    batch_size = 128
    lr = 0.1
    epochs = 100
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define data augmentation
    transformations = v2.Compose([
        v2.ToImage(),
        v2.Resize([96,96]),
        v2.ToDtype(torch.float32, scale=True)
    ])
    # Load the dataset
    train_dataset = FashionMNIST(dataset_root, train=True, transform=transformations)
    train_dataset, validation_dataset = random_split(train_dataset, [1-validation_ratio, validation_ratio])
    test_dataset = FashionMNIST(dataset_root, train=False, transform=transformations)
    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, 1, shuffle=False)
    # Initialize components
    net = GoogLeNet()
    opt = Adam(net.parameters(), lr=lr)
    loss = CrossEntropyLoss()
    # Training
    net,_,_,_,_ = train_network(net, train_loader, validation_loader, opt, loss, epochs, accuracy, device=device)
    # Testing
    test_network(net, test_loader, accuracy, device=device)
    
