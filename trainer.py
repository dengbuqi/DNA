"""
DNA v8 — Temporal Difference dopamine learning.
Phase 1: local Hebbian warm start
Phase 2: TD-dopamine (no target, only deviation from expected accuracy)
"""

import torch, gc
from torchvision import datasets, transforms
from model import Brain

args = {'batch_size': 256, 'phase1_epochs': 6, 'phase2_epochs': 10, 'lr': 0.01}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = '/home/deng/data/MNIST' if not torch.cuda.is_available() else '/data/MNIST'

def make_loader(train=True):
    return torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=train, download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,)), torch.nn.Flatten()])),
        batch_size=args['batch_size'], shuffle=train, num_workers=0)

def test(model, loader):
    cor = 0; tot = 0
    with torch.no_grad():
        for d, t in loader:
            d, t = d.to(device), t.to(device)
            s = model(d)
            cor += model.predict(s).eq(t).sum().item()
            tot += d.size(0)
    return cor / tot  # return as fraction, not percentage

if __name__ == '__main__':
    torch.cuda.empty_cache()
    train_loader = make_loader(True)
    test_loader = make_loader(False)

    model = Brain(in_features=784, out_features=10, cells_per_class=3, lr=0.01).to(device)
    print(f'{model.get_n_cells()} cells\n')

    # Phase 1
    print('=== Phase 1: Local Hebbian ===')
    for ep in range(args['phase1_epochs']):
        model.lr = 0.01; cor = 0; tot = 0
        for d, t in train_loader:
            d, t = d.to(device), t.to(device)
            s = model(d)
            model.phase1_update(d, t)
            cor += model.predict(s).eq(t).sum().item()
            tot += d.size(0)
        tr = cor / tot
        te = test(model, test_loader)
        print(f'  P1 ep{ep+1} Train: {tr*100:.2f}% Test: {te*100:.2f}%')

    best_p1 = test(model, test_loader)
    del train_loader; gc.collect(); torch.cuda.empty_cache()

    print(f'\nBest Phase 1: {best_p1*100:.2f}%')
    print('=== Phase 2: TD-Dopamine ===')

    train_loader2 = make_loader(True)
    # Initialize expected accuracy from Phase 1 result
    model.expected_accuracy = best_p1
    print(f'  Initial expected accuracy: {best_p1*100:.2f}%')

    for ep in range(args['phase2_epochs']):
        model.lr = 0.005; cor = 0; tot = 0; td_sum = 0.0; td_n = 0
        for d, t in train_loader2:
            d, t = d.to(device), t.to(device)
            with torch.no_grad():
                co = model.get_cell_out(d)
                model.update_eligibility(co)
                s = model(d)
                pred = model.predict(s)

                # Per-batch TD error
                batch_acc = pred.eq(t).float().mean().item()
                td_error = model.compute_td_error(batch_acc)
                model.apply_td_update(td_error)

                cor += pred.eq(t).sum().item()
                tot += d.size(0)
                td_sum += td_error
                td_n += 1
                del co, s
        torch.cuda.empty_cache()
        tr = cor / tot
        te = test(model, test_loader)
        td_avg = td_sum / td_n
        print(f'  P2 ep{ep+1} Train: {tr*100:.2f}% Test: {te*100:.2f}% '
              f'TD={td_avg:+.5f} ExpAcc={model.expected_accuracy*100:.2f}%')

    best_p2 = test(model, test_loader)
    print(f'\nPhase 1: {best_p1*100:.2f}%  Phase 2: {best_p2*100:.2f}%  Best: {max(best_p1, best_p2)*100:.2f}%')
