"""
DNA v3 Trainer — Overproduction + Energy Budget.

Training loop:
  1. Forward pass (BCE + backprop — standard PyTorch)
  2. Track activations per cell
  3. Epoch end: energy update, structural update (split rich, remove dead)
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import Brain


args = {
    'batch_size': 256,
    'epochs': 20,
    'lr': 0.01,
    'momentum': 0.5,
    'patience': 3,
    'initial_cells': 10,
    'max_cells': 32,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data/MNIST', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       torch.nn.Flatten(),
                   ])),
    batch_size=args['batch_size'], shuffle=True, num_workers=0)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data/MNIST', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       torch.nn.Flatten(),
                   ])),
    batch_size=args['test_batch_size'] if 'test_batch_size' in args else 1000,
    shuffle=False, num_workers=0)


def train_epoch(epoch, model, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, F.one_hot(target, 10).float())
        loss.backward()
        optimizer.step()

        # Track cell activations for energy update
        model.track_activations(data)

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total_samples += data.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100. * correct / total_samples
    print(f'[Epoch {epoch}] Loss: {avg_loss:.4f} Acc: {accuracy:.2f}%')
    return avg_loss


def test(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    return 100. * correct / total


if __name__ == '__main__':
    model = Brain(
        in_features=784, out_features=10,
        initial_cells=args['initial_cells'],
        max_cells=args['max_cells'],
    ).to(device)
    print(f'Initial alive cells: {model.get_n_cells()}')

    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

    for epoch in range(1, args['epochs'] + 1):
        train_epoch(epoch, model, optimizer)

        # Energy + structural update
        model.update_energy()
        model.structural_update()

        print(f'  [Arch] Alive cells: {model.get_n_cells()}')

        test_acc = test(model)
        print(f'  [Test] Acc: {test_acc:.2f}%')
        print()
