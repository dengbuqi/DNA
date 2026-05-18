"""
DNA v4 Trainer — No backpropagation, pure local learning, with lr annealing.
"""

import torch
from torchvision import datasets, transforms
from model import Brain


args = {
    'batch_size': 256,
    'epochs': 15,
    'lr': 0.01,
    'lr_decay': 0.5,
    'patience': 2,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Use local data path for WSL compatibility
DATA_PATH = '/home/deng/data/MNIST' if not torch.cuda.is_available() else '/data/MNIST'

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_PATH, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       torch.nn.Flatten(),
                   ])),
    batch_size=args['batch_size'], shuffle=True, num_workers=0)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_PATH, train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       torch.nn.Flatten(),
                   ])),
    batch_size=1000, shuffle=False, num_workers=0)


def test(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            scores = model(data)
            pred = model.predict(scores)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
    return 100. * correct / total


if __name__ == '__main__':
    model = Brain(
        in_features=784, out_features=10, cells_per_class=3,
        lr=args['lr'],
        base_cost=5.0,
        energy_init=100, energy_split=150,
    ).to(device)

    print(f'Alive cells: {model.get_n_cells()}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters())}')
    print()

    best_acc = 0
    plateau_count = 0
    current_lr = args['lr']

    for epoch in range(1, args['epochs'] + 1):
        model.train()
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            model.lr = current_lr
            scores = model(data)
            model.local_update(data, target)
            pred = model.predict(scores)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

        model.update_energy(data, target)
        model.structural_update()

        train_acc = 100. * correct / total
        test_acc = test(model)

        print(f'[Epoch {epoch}] Train: {train_acc:.2f}% Test: {test_acc:.2f}% '
              f'lr={current_lr:.5f} Alive={model.get_n_cells()}')

        if test_acc > best_acc:
            best_acc = test_acc
            plateau_count = 0
        else:
            plateau_count += 1
            if plateau_count >= args['patience']:
                current_lr *= args['lr_decay']
                plateau_count = 0

        print()

    print(f'Best test accuracy: {best_acc:.2f}%')
