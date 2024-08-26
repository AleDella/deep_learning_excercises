from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader,random_split
from torch.nn import CrossEntropyLoss
from torch.optim.adam import Adam
from torchvision.transforms import v2
import torch
from models import LinearRegressor
from metrics import accuracy
from training import train_network, test_network



if __name__ == '__main__':
    # Hyperparams
    dataset_root = "datasets\\FashionMNIST"
    validation_ratio = 0.3
    batch_size = 128
    input_size = 28*28
    output_dim = 10
    lr = 0.003
    epochs = 3
    # Define data augmentation
    transformations = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(lambda x: torch.flatten(x)) # Transformation to flatten the dataset
    ])
    # Load the dataset
    train_dataset = FashionMNIST(dataset_root, train=True, transform=transformations)
    train_dataset, validation_dataset = random_split(train_dataset, [1-validation_ratio, validation_ratio])
    test_dataset = FashionMNIST(dataset_root, train=False, transform=transformations)
    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, 1, shuffle=False)
    # Iniitialize components
    net = LinearRegressor(input_size, output_dim, True)
    opt = Adam(net.parameters(), lr=lr)
    loss = CrossEntropyLoss()
    # Training procedure
    net,_,_,_,_ = train_network(net, train_loader, validation_loader, opt, loss, epochs, accuracy, True)
    # Testing
    test_network(net, test_loader, accuracy)
    
        
