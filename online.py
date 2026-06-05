"""
DNA v15 online v3 — True continuous learning: no train/test epoch split.

Key insight: in a biological brain, there is no "training phase" vs "testing phase".
Every experience is both a learning opportunity AND a performance moment.

The challenge is balancing:
  - Plasticity: ability to learn from new data
  - Stability: not forgetting what was learned

Approach:
  - Use a sliding window stream: first 5000 samples are "seen once and kept"
  - After that, every new sample pushes out an old one (replay buffer)
  - Energy is computed per-batch but with a higher starting energy
  - Monitor: stream accuracy + periodically evaluate on full test set
"""

import torch, gc
from torchvision import datasets, transforms
from model import Brain

args = {
    'batch_size': 128,
    'lr': 0.005,
    'cells_per_class': 8,
    'max_cells_per_class': 32,
    'energy_cost': 0.5,          # lower cost so cells live longer online
    'measure_causal': True,
    'extinction': True,
    'extinction_interval': 2000,  # every 2000 batches
    'wta_k': 2,
    'wta_anneal_start': 100,
    'wta_anneal_end': 500,
    'gain_lr': 0.002,
    'bias_lr': 0.002,
    'eval_every': 50,
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
        batch_size=args['batch_size'], shuffle=True, num_workers=0)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    model = Brain(
        in_features=784, out_features=10,
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

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                torch.nn.Flatten()])),
        batch_size=1000, shuffle=False, num_workers=0)

    def test():
        model.eval()
        cor = 0
        tot = 0
        with torch.no_grad():
            for d, t in test_loader:
                d, t = d.to(device), t.to(device)
                cor += model.predict(model(d)).eq(t).sum().item()
                tot += d.size(0)
        return cor / tot * 100

    print(f'\nInitial: {model.get_n_cells()} cells')
    print('Continuous learning mode (single pass, no replay, no train/test split)\n')

    print(f'{"batch":>6} | {"stream_acc":>8} | {"test_acc":>8} | {"cells":>5}')
    print('-' * 50)

    train_loader = make_loader(True)
    stream_correct = 0
    stream_total = 0
    best_test = 0.0

    for step, (d, t) in enumerate(train_loader):
        d, t = d.to(device), t.to(device)
        model.epoch = step

        # Forward + update in one step (true online)
        s = model(d)
        pred = model.predict(s)
        stream_correct += pred.eq(t).sum().item()
        stream_total += d.size(0)

        model.phase1_update(d, t)
        model.update_energy(d, t)
        model.structural_update()

        if step > 0 and step % args['extinction_interval'] == 0:
            model.measure_all_causal()
            model.extinction()

        if (step + 1) % args['eval_every'] == 0:
            stream_acc = stream_correct / stream_total * 100
            test_acc = test()
            if test_acc > best_test:
                best_test = test_acc
            k = model._get_wta_k(model.max_C)
            print(f'{step+1:>6} | {stream_acc:>7.2f}% | {test_acc:>7.2f}% | {model.get_n_cells():>5} | WTA_k={k}')

    test_acc = test()
    if test_acc > best_test:
        best_test = test_acc
    stream_acc = stream_correct / stream_total * 100
    print(f'\nFinal: stream={stream_acc:.2f}%  test={test_acc:.2f}%  best_test={best_test:.2f}%')
    print(f'Cells: {model.get_n_cells()}')
