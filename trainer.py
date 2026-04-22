import torch
import numpy as np
import torch.nn as nn
from model import Brain
import torch.optim as optim
from datetime import datetime 
import torch.nn.functional as F
from torchvision import datasets, transforms

class FlattenTransform:
    def __call__(self, x):
        return torch.flatten(x)

def to_one_hot(labels, num_classes=10):
    return torch.nn.functional.one_hot(labels, num_classes).float()

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

    def __call__(self, val_loss, diff=0.01):
        if val_loss < self.best_loss-diff:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

def to_one_hot(labels, num_classes=10):
    return torch.nn.functional.one_hot(labels, num_classes).float()

args={}
kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
args['batch_size']=1000
args['test_batch_size']=1000
args['epochs']=20  # Increased epochs to allow for more extinction cycles
args['lr']=0.01 #Learning rate is how fast it will decend. 
args['momentum']=0.5 #SGD momentum (default: 0.5) Momentum is a moving average of our gradients (helps to keep direction).

args['seed']=1 #random seed
args['log_interval']=10
args['patience'] = 2 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load the data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data/MNIST', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        FlattenTransform()
                    ])),
    batch_size=args['batch_size'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data/MNIST', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                        FlattenTransform()
                    ])),
    batch_size=args['test_batch_size'], shuffle=True, **kwargs)

def train(epoch, model, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    for batch_idx, (data, target_raw) in enumerate(train_loader):
        data, target_raw = data.to(device, non_blocking=True), target_raw.to(device, non_blocking=True)
        target = to_one_hot(target_raw)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target_raw.view_as(pred)).sum().item()
        total_samples += data.size(0)
        
        early_stopping(loss.item())
        if early_stopping.early_stop:
            print(f'\n[Epoch {epoch}] Triggering Extinction Process...')
            model.extinction()
            optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
            early_stopping.reset()
            print('[Extinction Complete] Optimizer reset.\n')

    avg_loss = total_loss / total_samples
    accuracy = 100. * correct / total_samples
    print(f'[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')
def test():
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target_raw in test_loader:
            data, target_raw = data.to(device, non_blocking=True), target_raw.to(device, non_blocking=True)
            target = to_one_hot(target_raw)
            output = model(data)
            test_loss += F.binary_cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target_raw.view_as(pred)).sum().item()
            total_samples += data.size(0)

    avg_loss = test_loss / total_samples
    accuracy = 100. * correct / total_samples
    print(f'[Test] Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%\n')
    return avg_loss

if __name__ == '__main__':
    model = Brain(in_features=784, out_features=10)
    model._init()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

    early_stopping = EarlyStopping(patience=args['patience'], verbose=False)
    for epoch in range(1, args['epochs'] + 1):
        train(epoch,model,optimizer)
        val_loss = test()
