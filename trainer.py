"""
DNA v4 Trainer — No backpropagation, pure local learning.

Training loop:
  1. Forward pass (no grad)
  2. Local weight update (Δw = lr * x * (target - output))
  3. Energy update
  4. Structural update (kill/split)
  5. No loss.backward(), no optimizer
"""

import torch
from torchvision import datasets, transforms
from model import Brain


args = {
    'batch_size': 256,
    'epochs': 10,
    'lr': 0.01,
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
    ).to(device)

    print(f'Alive cells: {model.get_n_cells()}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters())}')
    print()

    for epoch in range(1, args['epochs'] + 1):
        model.train()
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward
            scores = model(data)

            # Local learning (no BP!)
            model.local_update(data, target)

            # Track accuracy
            pred = model.predict(scores)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

        # Energy + structural update
        model.update_energy(data, target)
        model.structural_update()

        print(f'[Epoch {epoch}] Train Acc: {100.*correct/total:.2f}% '
              f'Alive: {model.get_n_cells()}')

        test_acc = test(model)
        print(f'[Epoch {epoch}] Test Acc: {test_acc:.2f}%')
        print()
