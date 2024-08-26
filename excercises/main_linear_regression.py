# Imports
import torch
# import matplotlib.pyplot as plt
# Custom library
from models import LinearRegressor
from losses import squared_loss
from optimizers import sgd
from data_utils import synthetic_data,data_iter




if __name__ == '__main__':
    
    # Hyperparams
    lr = 0.01
    num_epochs = 3
    batch_size = 10
    
    # Generate the toy dataset
    true_w = torch.tensor([2, 3.4])
    true_b = torch.tensor(4.2)
    features, labels = synthetic_data(true_w, true_b, 1000)
    
    # Initialize the components
    net = LinearRegressor(len(true_w))
    loss = squared_loss
    opt = sgd(net.parameters(), lr)
    
    for epoch in range(num_epochs):
        train_l = []
        for X, y in data_iter(batch_size, features, labels):
            # Reset the grad
            opt.zero_grad()
            # Forward the net
            out = net(X)
            # Minibatch loss in `X` and `y`
            l = loss(out, y)
            # Backprop
            l.backward()
            # Optimization
            opt.step()
            # Save the losses 
            train_l.append(l.item())
        print(f'epoch {epoch + 1}, loss {float(sum(train_l)/len(train_l)):f}')

