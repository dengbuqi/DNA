"""
DNA v17 trainer — EML cells (exp(w₁·x) - ln(w₂·x)) + pure local Hebbian.
"""

import torch, gc
from torchvision import datasets, transforms
from model import Brain

args = {
    'batch_size': 256,
    'epochs': 50,
    'lr': 0.005,
    'lr_decay': 0.5,
    'patience': 4,
    'cells_per_class': 16,   # more cells for EML ensemble
    'max_cells_per_class': 64,
    'energy_cost': 0.5,
    'measure_causal': True,
    'extinction': True,
    'extinction_interval': 9999,  # no extinction for baseline test
    'extinction_mode': 'soft_energy',
    'extinction_keep': 12,
    'w1_scale': 0.1,    # exp branch (keep small)
    'w2_scale': 3.0,    # ln branch (higher for flatter gradient)
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = '/mnt/e/data/MNIST'
print(f'Device: {device}')
print(f'Config: {args}')


def make_loader(train=True):
    return torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                lambda x: x.flatten()])),
        batch_size=args['batch_size'], shuffle=train, num_workers=0)


def test(model, loader):
    model.eval()
    cor = 0
    tot = 0
    with torch.no_grad():
        for d, t in loader:
            d, t = d.to(device), t.to(device)
            s = model(d)
            cor += model.predict(s).eq(t).sum().item()
            tot += d.size(0)
    return cor / tot


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train_loader = make_loader(True)
    test_loader = make_loader(False)

    model = Brain(
        in_features=784, out_features=10,
        cells_per_class=args['cells_per_class'],
        max_cells_per_class=args['max_cells_per_class'],
        lr=args['lr'],
        base_cost=args['energy_cost'],
        extinction_mode=args['extinction_mode'],
        extinction_keep=args['extinction_keep'],
        w1_scale=args['w1_scale'],
        w2_scale=args['w2_scale'],
    ).to(device)

    total_cells = model.get_n_cells()
    print(f'\nInitial: {total_cells} cells = {total_cells / 10:.1f} per class\n')
    print(f'Cell type: EML (exp(w₁·x) - ln(w₂·x)) + sigmoid (no gain/bias, no WTA)\n')

    print('=== Training: EML Hebbian ===')
    best_test = 0.0
    plateau_count = 0
    current_lr = args['lr']

    for ep in range(args['epochs']):
        model.lr = current_lr
        cor = 0
        tot = 0
        for d, t in train_loader:
            d, t = d.to(device), t.to(device)
            s = model(d)
            model.phase1_update(d, t)
            cor += model.predict(s).eq(t).sum().item()
            tot += d.size(0)

        tr = cor / tot
        te = test(model, test_loader)
        if te > best_test:
            best_test = te
            plateau_count = 0
        else:
            plateau_count += 1
            if plateau_count >= args['patience']:
                current_lr *= args['lr_decay']
                plateau_count = 0

        if args['measure_causal']:
            model.measure_all_causal()
            model.update_energy(d, t)
            model.structural_update()
            print(f'  ep{ep+1:2d} Train: {tr*100:.2f}% Test: {te*100:.2f}% lr={current_lr:.5f} cells={model.get_n_cells()}')
        else:
            print(f'  ep{ep+1:2d} Train: {tr*100:.2f}% Test: {te*100:.2f}% lr={current_lr:.5f}')

        if args['extinction'] and (ep + 1) % args['extinction_interval'] == 0:
            model.epoch = ep
            model.extinction()

    print(f'\nBest test: {best_test*100:.2f}%')
    print(f'Final cells: {model.get_n_cells()}')
