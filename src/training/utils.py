from operator import itemgetter
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.nn import MSELoss

from metrics import log_rmse

def train_network(net, train_loader, validation_loader, opt, loss, epochs, evaluation_metric, print_res=True, device='cpu'):
    '''
    Function used to train a generic network in PyTorch
    
    Args:
        net (torch.nn.Module): Network to train
        train_loader, validation_loader (torch.utils.data.DataLoader): training and validation datasets
        opt (torch.optim): Optimizer for the parameters
        loss (torch.nn._Loss): Loss to use
        epochs (int): number of epochs to train for
        evaluation_metric (func): function for the evaluation score
        print_res (bool): flag to print the results while training
        device (str): device for the training
    Returns:
        net (torch.nn.Module): trained network
    '''
    # Load the network into the device
    net = net.to(device)
    for i in range(epochs):
        print(f"Epoch {i+1} ------------------------------------------")
        # Save the training losses
        net.train()
        train_losses = []
        train_accuracy = []
        # Train cycle
        for images,labels in train_loader:
            # Load the data into the device
            images = images.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            output = net(images)
            l = loss(output, labels)
            l.backward()
            opt.step()
            acc = evaluation_metric(output,labels)
            train_accuracy.append(acc)
            train_losses.append(l.item())
        if print_res:
            print(f"Train Loss: {sum(train_losses)/len(train_loader)}\tTrain Accuracy: {sum(train_accuracy)/len(train_loader)}")
        # Save the validation losses
        net.eval()
        val_losses = []
        val_accuracy = []
        # Validation cycle
        for images,labels in validation_loader:
            # Load the data into the device
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            l = loss(output, labels)
            acc = evaluation_metric(output,labels)
            val_accuracy.append(acc)
            val_losses.append(l.item())
        if print_res:
            print(f"Validation Loss: {sum(val_losses)/len(validation_loader)}\tValidation Accuracy: {sum(val_accuracy)/len(validation_loader)}")
        
    return net, sum(train_losses)/len(train_loader), sum(train_accuracy)/len(train_loader), sum(val_losses)/len(validation_loader), sum(val_accuracy)/len(validation_loader)
    
    
def test_network(net, test_loader, evaluation_metric, device='cpu'):
    '''
    Function used to test a generic network in PyTorch
    
    Args:
        net (torch.nn.Module): Network to train
        test_loader (torch.utils.data.DataLoader): test datasets
        evaluation_metric (func): function for the evaluation score
        device (str): device for the training
    '''
    # Load the network into the device
    net = net.to(device)
    # Testing
    net.eval()
    test_accuracy = []
    # Test cycle
    for images,labels in test_loader:
        # Load the data into the device
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)
        acc = evaluation_metric(output,labels)
        test_accuracy.append(acc)
    print(f"Test Accuracy: {sum(test_accuracy)/len(test_loader)}")
    
def k_fold_training(net_func, training_set, net_params={}, opt_func=Adam, loss_func=MSELoss, epochs=10, batch_size=128, lr=0.01, weight_decay=0, evaluation_metric=log_rmse, n_splits=5, print_res=True, device='cpu'):
    '''
    Function used to train a generic network in PyTorch
    
    Args:
        net_func (torch.nn.Module): Network to initialize for training
        training_set (iterable): training dataset
        net_params (dict): initialization parameters for the network
        opt_func (torch.optim): Optimizer function to use
        loss_func (torch.nn._Loss): Loss function to use
        epochs (int): number of epochs to train for
        batch_size (int): number of samples for each forward of the network
        lr (float): learning rate
        weight_decay (float): weight decay rate
        evaluation_metric (func): function for the evaluation score
        n_splits (int): number of splits for KFold cross validation
        print_res (bool): flag to print the results while training
        device (str): device for the training
    Returns:
        net (torch.nn.Module): trained network
    '''
    kfold = KFold(n_splits, shuffle=True)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(training_set)):
        # Get current fold's train and validation set
        train_set = itemgetter(*train_ids)(training_set)
        val_set = itemgetter(*val_ids)(training_set)
        # Create the dataloader
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        # Initialize components for training
        net = net_func(**net_params)
        opt = opt_func(net.parameters(), lr=lr, weight_decay=weight_decay)
        loss = loss_func()
        if print_res:
            print(f"============ Fold {fold+1} ============")
        fold_train_losses = []
        fold_train_accuracies = []
        fold_val_losses = []
        fold_val_accuracies = []
        net, train_loss, train_acc, val_loss, val_acc = train_network(net, train_loader, val_loader, opt, loss, epochs, evaluation_metric, print_res, device)
        fold_train_losses.append(train_loss)
        fold_train_accuracies.append(train_acc)
        fold_val_losses.append(val_loss)
        fold_val_accuracies.append(val_acc)
        if print_res:
            print(f"============ Fold Statistics ============")
            print(f"Train Loss: {sum(fold_train_losses)/len(fold_train_losses)}\tTrain Accuracy: {sum(fold_train_accuracies)/len(fold_train_accuracies)}")
            print(f"Validation Loss: {sum(fold_val_losses)/len(fold_val_losses)}\tValidation Accuracy: {sum(fold_val_accuracies)/len(fold_val_accuracies)}")
        