import torch
import numpy as np
import torch.nn as nn
from model import Brain
import torch.optim as optim
from datetime import datetime 
import torch.nn.functional as F
from torchvision import datasets, transforms
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def reset(self):
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

args={}
kwargs={}
args['batch_size']=1000
args['test_batch_size']=1000
args['epochs']=10  #The number of Epochs is the number of times you go through the full dataset. 
args['lr']=0.01 #Learning rate is how fast it will decend. 
args['momentum']=0.5 #SGD momentum (default: 0.5) Momentum is a moving average of our gradients (helps to keep direction).

args['seed']=1 #random seed
args['log_interval']=10
args['patience'] = 5 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load the data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data/MNIST', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        transforms.Lambda(lambda x: torch.flatten(x))
                    ])),
    batch_size=args['batch_size'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data/MNIST', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        transforms.Lambda(lambda x: torch.flatten(x))
                    ])),
    batch_size=args['test_batch_size'], shuffle=True, **kwargs)

def train(epoch,model,optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #Variables in Pytorch are differenciable. 
        #This will zero out the gradients for this batch. 
        optimizer.zero_grad()
        output = model(data)
        # Calculate the loss The negative log likelihood loss. It is useful to train a classification problem with C classes.
        loss = F.nll_loss(output, target)
        #dloss/dx for every Variable 
        loss.backward()
        #to do a one-step update on our parameter.
        optimizer.step()
        #Print out the loss periodically. 
        if batch_idx % args['log_interval'] == 0:
            print('{}|Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(datetime.now(), epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n{}|Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(datetime.now(), test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss

model = Brain(in_features=784, out_features=10).to(device)
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

early_stopping = EarlyStopping(patience=args['patience'], verbose=False)
for epoch in range(1, args['epochs'] + 1):
    train(epoch,model,optimizer)
    test()
    val_loss = test()
    early_stopping(val_loss)
    if early_stopping.early_stop:
        model.extinction()
        optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
