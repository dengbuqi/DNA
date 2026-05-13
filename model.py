import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


# --- Constants ---
MAX_CELLS_PER_EDGE = 16
MIN_HIDDEN_SIZE = 4


class NeuralCell(nn.Module):
    """A single neural cell: linear mapping + vitality tracking + dying state."""

    def __init__(self, in_features=1, out_features=1, bias=True):
        super().__init__()
        self.f = nn.Linear(in_features, out_features, bias=bias)

        # --- Vitality Tracking (Python floats, no GPU sync issues) ---
        self.grad_norm = 0.0
        self.activation_sum = 0.0
        self.sample_count = 0

        # --- Dying State ---
        self.dying = False
        self.decay_factor = 1.0

        # --- Grace Period ---
        self.grace_remaining = 0

    def _init(self, w=0.01):
        nn.init.constant_(self.f.weight, w)
        nn.init.zeros_(self.f.bias)

    def setweight(self, weight, bias):
        with torch.no_grad():
            self.f.weight.copy_(weight)
            self.f.bias.copy_(bias)

    def getweight(self):
        return self.f.weight, self.f.bias

    @property
    def vitality(self):
        if self.sample_count == 0:
            return 0.0
        avg_activation = self.activation_sum / max(self.sample_count, 1)
        return self.grad_norm * avg_activation

    def reset_vitality(self):
        self.grad_norm = 0.0
        self.activation_sum = 0.0
        self.sample_count = 0

    def track_forward(self, output):
        self.activation_sum += output.abs().sum().item()
        self.sample_count += output.numel()

    def track_backward(self):
        total_norm = 0.0
        for p in self.f.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        self.grad_norm += total_norm ** 0.5

    def enter_dying(self):
        self.dying = True
        self.decay_factor = 0.5

    def recover_from_dying(self):
        self.dying = False
        self.decay_factor = 1.0

    def decay(self):
        if self.dying:
            self.decay_factor *= 0.5

    def is_dead(self):
        return self.dying and self.decay_factor < 0.01

    def forward(self, x):
        out = self.f(x)
        if self.dying:
            out = out * self.decay_factor
        return out


class NeuralCellEdge(nn.Module):
    """
    A bundle of parallel NeuralCells connecting one input feature to all outputs.

    Flat mode:      Input [1] ---> parallel NeuralCells [1->out] ---> sum/n ---> Output [out]
    Hierarchical:   Input [1] ---> hidden [1->H] -> ReLU ---> parallel NeuralCells [H->out] ---> sum/n ---> Output [out]
    """

    def __init__(self, in_features=1, out_features=1, bias=True, init_w=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.fs = nn.ModuleList()
        self.hidden_layer = None
        self.device = 'cpu'

        if init_w is not None:
            self._init(init_w)

    @property
    def is_hierarchical(self):
        return self.hidden_layer is not None

    def _init(self, w):
        cell = NeuralCell(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias
        )
        self.fs.append(cell)
        cell._init(w)

    def _grow_hidden_layer(self):
        N = len(self.fs)
        H = max(MIN_HIDDEN_SIZE, N // 2)

        hidden = nn.Linear(self.in_features, H, bias=True)
        nn.init.normal_(hidden.weight, mean=0.0, std=0.1)
        nn.init.zeros_(hidden.bias)
        self.hidden_layer = hidden.to(self.device)

        new_cells = nn.ModuleList()
        for old_cell in self.fs:
            old_w, old_b = old_cell.getweight()
            new_cell = NeuralCell(in_features=H, out_features=self.out_features, bias=self.bias).to(self.device)
            new_w = torch.zeros(self.out_features, H, device=self.device)
            new_w[:, 0:1] = old_w
            new_cell.setweight(new_w, old_b)
            new_cells.append(new_cell)

        self.fs = new_cells
        print(f'  [Growth] Edge {id(self) % 1000}: flat -> hierarchical '
              f'({self.in_features}->{H}->{self.out_features}, {N} cells)')

    def to(self, device):
        self.device = device
        for f in self.fs:
            f.to(device)
        if self.hidden_layer is not None:
            self.hidden_layer = self.hidden_layer.to(device)
        return self

    def __str__(self):
        return f'{len(self.fs)}' + ('(H)' if self.is_hierarchical else '')

    def reset_vitality(self):
        for f in self.fs:
            f.reset_vitality()

    def get_vitalities(self):
        return torch.tensor([f.vitality for f in self.fs], device=self.device)

    def structural_update(self, vitality_threshold):
        # Phase 0: Grow hidden layer if needed
        if not self.is_hierarchical and len(self.fs) > MAX_CELLS_PER_EDGE:
            self._grow_hidden_layer()

        # Phase 1: Split high-vitality cells
        vitals = self.get_vitalities()
        create_indices = torch.nonzero(vitals > vitality_threshold * 2, as_tuple=True)[0]
        if len(create_indices) > 0:
            create_sorted = create_indices[torch.argsort(vitals[create_indices], descending=True)]
            # Only split top half of qualifying cells to control growth
            max_split = max(1, len(create_sorted) // 2)
            for idx in create_sorted[:max_split].tolist():
                self._create_cell(idx)

        # Phase 2: Dying/recovery management
        vitals_after = self.get_vitalities()
        for i, f in enumerate(self.fs):
            if f.grace_remaining > 0:
                f.grace_remaining -= 1
                continue
            vitality_val = vitals_after[i].item()
            if f.dying:
                if vitality_val >= vitality_threshold:
                    f.recover_from_dying()
                else:
                    f.decay()
            else:
                if vitality_val < vitality_threshold:
                    f.enter_dying()

        # Phase 3: Remove fully dead cells
        self._cleanup_dead()

    def _create_cell(self, idx):
        try:
            newf = NeuralCell(
                in_features=self.fs[idx].f.in_features,
                out_features=self.out_features,
                bias=self.bias,
            ).to(self.device)
            nn.init.normal_(newf.f.weight, mean=0.0, std=0.01)
            nn.init.zeros_(newf.f.bias)
            newf.grace_remaining = 1
            self.fs.insert(idx + 1, newf)
        except Exception as e:
            print(f'Create cell at {idx} failed:', e)

    def _cleanup_dead(self):
        alive = [f for f in self.fs if not f.is_dead()]
        if len(alive) > 0:
            self.fs = nn.ModuleList(alive)

    def forward(self, x):
        if self.is_hierarchical:
            h = F.relu(self.hidden_layer(x))
        else:
            h = x

        n_cells = len(self.fs)
        if n_cells == 0:
            return torch.zeros(x.shape[0], self.out_features, device=x.device)

        y = 0
        for f in self.fs:
            out = f(h)
            if not f.dying:
                f.track_forward(out)
            y += out

        return y / n_cells


class Brain(nn.Module):
    """The Brain: complete DNA model with structural plasticity."""

    def __init__(self, in_features=784, out_features=10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_w = 1.0
        self.vitality_threshold = 0.5
        self.edges = nn.ModuleList()
        self.device = 'cpu'

    def _init(self):
        self.edges = nn.ModuleList()
        for _ in range(self.in_features):
            self.edges.append(
                NeuralCellEdge(1, self.out_features, bias=True, init_w=self.init_w)
            )

    def to(self, device):
        self.device = device
        for es in self.edges:
            es.to(device)
        return self

    def reset_vitality(self):
        for es in self.edges:
            es.reset_vitality()

    def track_backward(self):
        for es in self.edges:
            for f in es.fs:
                f.track_backward()

    def structural_update(self):
        for es in self.edges:
            es.structural_update(self.vitality_threshold)

    def extinction(self):
        print('[Extinction] Triggered. Marking low-vitality cells...')
        for es in self.edges:
            vitals = es.get_vitalities()
            for i, f in enumerate(es.fs):
                if f.grace_remaining > 0:
                    continue
                if not f.dying and vitals[i].item() < self.vitality_threshold * 2:
                    f.enter_dying()
            es._cleanup_dead()

        self.vitality_threshold /= 2
        print(f'[Extinction] Vitality threshold halved to {self.vitality_threshold:.6f}')

    def _get_arch(self):
        cell_counts = np.array([len(es.fs) for es in self.edges])
        hier_count = sum(1 for es in self.edges if es.is_hierarchical)
        return cell_counts, hier_count

    def __str__(self):
        cell_counts, hier_count = self._get_arch()
        table = pd.DataFrame(cell_counts.reshape(-1, 1), columns=['cells'])
        dying_count = sum(1 for es in self.edges for f in es.fs if f.dying)
        grace_count = sum(1 for es in self.edges for f in es.fs if f.grace_remaining > 0)
        total_cells = cell_counts.sum()
        return (
            f'DNA Model: {total_cells} cells ({dying_count} dying, {grace_count} grace) '
            f'| {hier_count} hierarchical edges\n'
            f'Edge cell distribution:\n{table.to_string()}'
        )

    def forward(self, x, act=F.sigmoid):
        batch_size = x.shape[0]
        ys = torch.zeros(batch_size, self.out_features, device=x.device)

        for i, edge in enumerate(self.edges):
            inp = x[:, i:i+1]
            ys += edge(inp)

        return act(ys)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = Brain(in_features=784, out_features=10)
    m._init()
    m.to(device)
    x = torch.rand(10, 784).to(device)
    y = m(x)
    print(m)
    print(f'Output shape: {y.shape}')

    print('\n--- Simulating growth: force one edge to exceed MAX_CELLS_PER_EDGE ---')
    edge0 = m.edges[0]
    for _ in range(MAX_CELLS_PER_EDGE + 1):
        edge0._create_cell(0)

    print(f'Edge 0 before structural_update: {len(edge0.fs)} cells')
    edge0.structural_update(m.vitality_threshold)
    print(f'Edge 0 after structural_update: {len(edge0.fs)} cells, hierarchical={edge0.is_hierarchical}')

    print(f'\nTest forward pass after growth:')
    y = m(x)
    print(f'Output shape: {y.shape}')
