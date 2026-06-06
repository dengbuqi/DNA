"""
DNA online — 连续学习 + replay buffer.
"""

import torch, gc
import torch.nn.functional as F
from torchvision import datasets, transforms
from model import Brain
import random

args = {
    'batch_size': 128,
    'lr': 0.005,
    'lr_decay': 0.97,
    'cells_per_class': 8,
    'max_cells_per_class': 32,
    'energy_cost': 0.5,
    'measure_causal': True,
    'extinction': True,
    'extinction_interval': 5000,
    'wta_k': 1,
    'wta_anneal_start': 150,
    'wta_anneal_end': 500,
    'gain_lr': 0.002,
    'bias_lr': 0.002,
    'eval_every': 100,
    'replay_ratio': 0.5,
    'replay_buffer_size': 20000,
    'total_updates': 30000,
    'extinction_mode': 'topk',
    'extinction_keep': 12,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = '/mnt/e/data/MNIST'
print(f'Device: {device}')
print(f'Config: {args}')


def make_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        torch.nn.Flatten()])

    train_dataset = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_PATH, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False, num_workers=0)

    return train_loader, test_loader, train_dataset, test_dataset


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
        extinction_mode=args['extinction_mode'],
        extinction_keep=args['extinction_keep'],
    ).to(device)

    train_loader, test_loader, train_dataset, _ = make_loaders()

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

    # Build replay buffer (images stored as flat tensors)
    replay_size = min(args['replay_buffer_size'], len(train_dataset))
    all_indices = list(range(len(train_dataset)))
    random.shuffle(all_indices)
    replay_buffer = []
    for i in all_indices[:replay_size]:
        x, y = train_dataset[i]
        replay_buffer.append((x.flatten(), y))

    print(f'\nInitial: {model.get_n_cells()} cells')
    print(f'Replay buffer: {len(replay_buffer)} samples')
    print('Continuous learning with replay\n')

    print(f'{"step":>6} | {"stream_acc":>8} | {"test_acc":>8} | {"cells":>5}')
    print('-' * 50)

    # Pre-load all new data as a big shuffled list
    new_data = []
    for d, t in train_loader:
        for i in range(len(d)):
            new_data.append((d[i].flatten(), t[i]))
    random.shuffle(new_data)

    stream_correct = 0
    stream_total = 0
    best_test = 0.0
    current_lr = args['lr']
    replay_cursor = 0
    new_cursor = 0

    for step in range(args['total_updates']):
        model.epoch = step
        model.lr = current_lr
        B = args['batch_size']

        # Build batch
        batch_x = []
        batch_y = []

        n_replay = int(B * args['replay_ratio'])
        n_new = B - n_replay

        for _ in range(n_replay):
            x, y = replay_buffer[replay_cursor % len(replay_buffer)]
            batch_x.append(x)
            batch_y.append(y)
            replay_cursor += 1

        for _ in range(n_new):
            x, y = new_data[new_cursor % len(new_data)]
            batch_x.append(x)
            batch_y.append(y)
            new_cursor += 1

        d = torch.stack(batch_x).to(device)
        t = torch.tensor(batch_y, device=device)

        # Forward + update
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
            else:
                current_lr *= args['lr_decay']
            k = model._get_wta_k(model.max_C)
            print(f'{step+1:>6} | {stream_acc:>7.2f}% | {test_acc:>7.2f}% | {model.get_n_cells():>5} | WTA_k={k}')

    test_acc = test()
    if test_acc > best_test:
        best_test = test_acc
    stream_acc = stream_correct / stream_total * 100
    print(f'\nFinal: stream={stream_acc:.2f}%  test={test_acc:.2f}%  best_test={best_test:.2f}%')
    print(f'Cells: {model.get_n_cells()}')
