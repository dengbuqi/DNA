import torch
from torchvision import datasets, transforms
from model import Brain

args = {'batch_size': 256, 'phase1_epochs': 6, 'phase2_epochs': 6, 'lr': 0.01}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = '/home/deng/data/MNIST' if not torch.cuda.is_available() else '/data/MNIST'

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_PATH, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
                       torch.nn.Flatten()])),
    batch_size=args['batch_size'], shuffle=True, num_workers=0)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_PATH, train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
                       torch.nn.Flatten()])),
    batch_size=1000, shuffle=False, num_workers=0)


def test(model):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for d, t in test_loader:
            d, t = d.to(device), t.to(device)
            s = model(d)
            correct += model.predict(s).eq(t).sum().item()
            total += d.size(0)
    return 100. * correct / total


if __name__ == '__main__':
    torch.cuda.empty_cache()

    model = Brain(in_features=784, out_features=10, cells_per_class=3, lr=0.01).to(device)
    print(f'{model.get_n_cells()} cells\n')

    # Phase 1
    print('=== Phase 1: Local Hebbian ===')
    lr = 0.01
    for ep in range(args['phase1_epochs']):
        model.lr = lr
        cor = 0; tot = 0
        for d, t in train_loader:
            d, t = d.to(device), t.to(device)
            s = model(d)
            model.phase1_update(d, t)
            cor += model.predict(s).eq(t).sum().item()
            tot += d.size(0)
        tr = 100. * cor / tot
        te = test(model)
        if te < test(model): pass  # decay logic placeholder
        print(f'  P1 ep{ep+1} Train: {tr:.2f}% Test: {te:.2f}%')

    best_p1 = te
    torch.cuda.empty_cache()

    # Phase 2
    print('\n=== Phase 2: Pure dopamine ===')
    lr = 0.005
    for ep in range(args['phase2_epochs']):
        model.lr = lr
        cor = 0; tot = 0; dop_avg = 0.0
        for d, t in train_loader:
            d, t = d.to(device), t.to(device)
            s = model(d)
            co = model.get_cell_out(d)
            model.update_eligibility(co)
            pred = model.predict(s)
            dopamine = (pred == t).float().mean().item() * 2 - 1
            model.apply_dopamine(dopamine)
            cor += pred.eq(t).sum().item()
            tot += d.size(0)
            dop_avg += dopamine
            del co, s, pred
        torch.cuda.empty_cache()
        dop_avg /= len(train_loader)
        tr = 100. * cor / tot
        te = test(model)
        print(f'  P2 ep{ep+1} Train: {tr:.2f}% Test: {te:.2f}% dop={dop_avg:+.3f}')

    best_p2 = te
    print(f'\nPhase 1: {best_p1:.2f}%  Phase 2: {best_p2:.2f}%  Best: {max(best_p1, best_p2):.2f}%')
