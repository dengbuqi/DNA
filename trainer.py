"""
DNA v15 trainer — Nonlinear cells + WTA + Hebbian gain/bias.
"""

import torch, gc
from torchvision import datasets, transforms
from model import Brain

# --- Config ---
args = {
    'batch_size': 256,
    'epochs': 30,
    'lr': 0.01,
    'lr_decay': 0.5,
    'patience': 3,
    'cells_per_class': 8,
    'max_cells_per_class': 32,
    'energy_cost': 2.0,
    'measure_causal': True,
    'extinction': True,
    'extinction_interval': 10,
    'wta_k': 1,
    'wta_anneal_start': 5,
    'wta_anneal_end': 20,
    'gain_lr': 0.005,
    'bias_lr': 0.005,
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
                torch.nn.Flatten()])),
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
        in_features=784,
        out_features=10,
        cells_per_class=args['cells_per_class'],
        max_cells_per_class=args['max_cells_per_class'],
        lr=args['lr'],
        base_cost=args['energy_cost'],
        wta_k=args['wta_k'],
        wta_anneal_start=args['wta_anneal_start'],
        wta_anneal_end=args['wta_anneal_end'],
        gain_lr=args['gain_lr'],
        bias_lr=args['bias_lr'],
    ).to(device)

    total_cells = model.get_n_cells()
    print(f'\nInitial: {total_cells} cells = {total_cells / 10:.1f} per class\n')

    print('=== Training: Hebbian + Nonlinear + WTA ===')
    best_test = 0.0
    plateau_count = 0
    current_lr = args['lr']

    for ep in range(args['epochs']):
        model.lr = current_lr
        model.epoch = ep
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

        k = model._get_wta_k(model.max_C)
        if args['measure_causal']:
            model.measure_all_causal()
            model.update_energy(d, t)
            model.structural_update()
            print(f'  ep{ep+1:2d} Train: {tr*100:.2f}% Test: {te*100:.2f}% lr={current_lr:.5f} cells={model.get_n_cells()} WTA_k={k}')
        else:
            print(f'  ep{ep+1:2d} Train: {tr*100:.2f}% Test: {te*100:.2f}% lr={current_lr:.5f} WTA_k={k}')

        if args['extinction'] and (ep + 1) % args['extinction_interval'] == 0:
            model.extinction()

    print(f'\nBest test: {best_test*100:.2f}%')
    print(f'Final cells: {model.get_n_cells()}')
