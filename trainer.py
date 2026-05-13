import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import Brain


class FlattenTransform:
    def __call__(self, x):
        return torch.flatten(x)


class EarlyStopping:
    """Used in reverse: trigger extinction when loss plateaus"""

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
        if val_loss < self.best_loss - diff:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'  Plateau counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


# --- Hyperparameters ---
args = {
    'batch_size': 256,
    'test_batch_size': 1000,
    'epochs': 30,
    'lr': 0.01,
    'momentum': 0.5,
    'seed': 1,
    'log_interval': 10,
    'patience': 3,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# --- Data Loaders ---
kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data/MNIST', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       FlattenTransform()
                   ])),
    batch_size=args['batch_size'], shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/data/MNIST', train=False,
                   transform=transforms.Compose([
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
        target = F.one_hot(target_raw, 10).float()

        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()

        # Track gradient norms for vitality calculation
        model.track_backward()

        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target_raw.view_as(pred)).sum().item()
        total_samples += data.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100. * correct / total_samples
    print(f'[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')
    return avg_loss


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target_raw in test_loader:
            data, target_raw = data.to(device, non_blocking=True), target_raw.to(device, non_blocking=True)
            output = model(data)
            loss = F.binary_cross_entropy(
                output, F.one_hot(target_raw, 10).float(), reduction='mean'
            )
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target_raw.view_as(pred)).sum().item()
            total_samples += data.size(0)

    avg_loss = test_loss / total_samples
    accuracy = 100. * correct / total_samples
    print(f'[Test]  Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')
    return avg_loss


if __name__ == '__main__':
    model = Brain(in_features=784, out_features=10)
    model._init()
    model.to(device)
    first_total = sum(len(es.fs) for es in model.edges)
    print(f'Initial model: {first_total} cells ({model.in_features} edges x 1 cell)')
    print()

    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
    early_stopping = EarlyStopping(patience=args['patience'], verbose=False)

    for epoch in range(1, args['epochs'] + 1):
        # Phase 1: Reset vitality stats for this epoch
        model.reset_vitality()

        # Phase 2: Train
        train_loss = train(epoch, model, optimizer)

        # Phase 3: Evaluate
        val_loss = test(model)

        # Phase 4: Structural update based on vitality
        model.structural_update()

        # Phase 5: Check extinction trigger (loss plateau)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f'\n  *** Loss plateau detected! Triggering Extinction... ***')
            model.extinction()
            optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
            early_stopping.reset()
            print()

        # Log architecture summary
        total_cells = sum(len(es.fs) for es in model.edges)
        dying_cells = sum(1 for es in model.edges for f in es.fs if f.dying)
        grace_cells = sum(1 for es in model.edges for f in es.fs if f.grace_remaining > 0)
        hier_edges = sum(1 for es in model.edges if es.is_hierarchical)
        avg_per_edge = total_cells / max(len(model.edges), 1)
        print(f'  [Arch] {total_cells} cells ({dying_cells} dying, {grace_cells} grace) | '
              f'{hier_edges} hierarchical | Avg {avg_per_edge:.2f} cells/edge | '
              f'Threshold {model.vitality_threshold:.6f}')
        print()
