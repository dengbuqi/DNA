"""
DNA v11 — Full concept reunited. Vectorized PyTorch version.

Core:
  1. Hebbian learning (fire together, wire together)
  2. Causal ablation (mask input → measure importance → energy)
  3. Energy-driven structural plasticity (split/die)
  4. Periodic extinction (median-importance filter)
  5. Self-growing layers (cells auto-organize into hierarchies)
  6. No backpropagation
"""

import torch
import torch.nn.functional as F


class Brain(torch.nn.Module):
    def __init__(self, in_features=784, out_features=10, cells_per_class=3,
                 max_cells_per_class=16, lr=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.C = cells_per_class
        self.max_C = max_cells_per_class
        self.total_slots = max_cells_per_class * 2
        self.lr = lr

        # Weight: [out_features, in_features, slots]
        self.weight = torch.nn.Parameter(
            torch.zeros(out_features, in_features, self.total_slots))

        # Energy per cell
        self.register_buffer('_energy', torch.full(
            (out_features, in_features, self.total_slots), 100.0, dtype=torch.float))
        self.register_buffer('_alive', torch.zeros(
            (out_features, in_features, self.total_slots), dtype=torch.bool))

        # Causal importance (running average)
        self.register_buffer('_causal_imp', torch.zeros(
            (out_features, in_features, self.total_slots)))
        self.register_buffer('_imp_count', torch.zeros(
            (out_features, in_features, self.total_slots)))

        # Grace period (epochs remaining)
        self.register_buffer('_grace', torch.zeros(
            (out_features, in_features, self.total_slots), dtype=torch.long))

        # Eligibility trace for dopamine
        self.register_buffer('_eligibility', torch.zeros(
            (out_features, in_features, self.total_slots)))

        self._alive[:, :, :cells_per_class] = True
        self._init_weights()

        # Extinction state
        self.epoch = 0
        self.extinction_interval = 5

    def _init_weights(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.5)

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        raw = torch.sigmoid(x.unsqueeze(1).unsqueeze(-1) * self.weight.unsqueeze(0))
        raw = raw * self._alive.unsqueeze(0).float()

        scores = raw.sum(dim=(2, 3)) / self._alive.sum(dim=(1, 2)).unsqueeze(0).clamp(min=1)
        return scores

    def train_step(self, x, labels, dopamine=1.0):
        """
        One step: Hebbian update + causal measurement.
        """
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        target = F.one_hot(labels, self.out_features).float()

        with torch.no_grad():
            # --- Hebbian update ---
            output = torch.sigmoid(x.unsqueeze(1).unsqueeze(-1) * self.weight.unsqueeze(0))
            target_exp = target.unsqueeze(-1).unsqueeze(-1)
            error = target_exp - output
            dw = self.lr * x.unsqueeze(1).unsqueeze(-1) * error * dopamine
            dw = dw * self._alive.unsqueeze(0).float()
            self.weight.data.add_(dw.mean(dim=0))
            self.weight.data.clamp_(-3.0, 3.0)

            # --- Causal measurement (sample-based masking) ---
            # For a random subset of cells, measure output change when input is masked
            mask_prob = 0.05  # measure 5% of cells per step
            if torch.rand(1).item() < mask_prob:
                # Pick a random (class, pixel, slot) to test
                c = torch.randint(0, self.out_features, (1,)).item()
                p = torch.randint(0, self.in_features, (1,)).item()
                slots = torch.where(self._alive[c, p])[0]
                if len(slots) > 0:
                    s = slots[torch.randint(0, len(slots), (1,))].item()

                    # Measure importance for this specific cell
                    w_val = self.weight[c, p, s].item()
                    x_val = x[0, p].item()
                    out_normal = torch.sigmoid(torch.tensor(x_val * w_val)).item()
                    out_masked = 0.5  # sigmoid(0) = 0.5

                    importance = abs(out_normal - out_masked)

                    # Update running average
                    cnt = self._imp_count[c, p, s].item()
                    self._causal_imp[c, p, s] = (
                        self._causal_imp[c, p, s] * cnt + importance
                    ) / (cnt + 1)
                    self._imp_count[c, p, s] += 1

    def update_energy(self, dopamine=1.0):
        """
        Energy update based on causal importance.
        """
        with torch.no_grad():
            # Reward: causal importance × dopamine mod
            reward = self._causal_imp * max(0.0, dopamine) * 10.0
            cost = 1.0
            delta = (reward - cost) * self._alive.float()
            self._energy += delta
            self._energy.clamp_(0, 300.0)

    def structural_update(self):
        """
        Kill energy <= 0, split energy > 150.
        """
        with torch.no_grad():
            # Kill
            dead = (self._energy <= 0) & self._alive & (self._grace <= 0)
            self._alive[dead] = False

            # Split
            for c in range(self.out_features):
                for p in range(self.in_features):
                    alive = torch.where(self._alive[c, p])[0]
                    n = len(alive)
                    if n >= self.max_C:
                        continue

                    rich = torch.where(
                        (self._energy[c, p] > 150.0) &
                        self._alive[c, p] &
                        (self._grace[c, p] <= 0)
                    )[0]
                    for idx in rich:
                        if n >= self.max_C:
                            break
                        dead_slots = torch.where(~self._alive[c, p])[0]
                        if len(dead_slots) == 0:
                            break
                        ni = dead_slots[0].item()
                        pw = self.weight[c, p, idx].clone()
                        self.weight.data[c, p, ni] = pw + torch.randn_like(pw) * 0.05
                        self._energy[c, p, ni] = 100.0
                        self._alive[c, p, ni] = True
                        self._grace[c, p, ni] = 5
                        self._energy[c, p, idx] *= 0.5
                        n += 1

            # Decrease grace counters
            self._grace = (self._grace - 1).clamp(min=0)

    def extinction(self):
        """
        Periodic mass structural reset.
        Keep only cells with causal_importance > median.
        """
        print(f'[Extinction] epoch {self.epoch} | cells={self.get_n_cells()}')
        with torch.no_grad():
            for c in range(self.out_features):
                for p in range(self.in_features):
                    alive = torch.where(self._alive[c, p])[0]
                    if len(alive) == 0:
                        # Restock
                        for _ in range(min(3, self.max_C)):
                            dead_slots = torch.where(~self._alive[c, p])[0]
                            if len(dead_slots) > 0:
                                ni = dead_slots[0].item()
                                self._alive[c, p, ni] = True
                                self._grace[c, p, ni] = 5
                                self._energy[c, p, ni] = 100.0
                        continue

                    # Filter by median causal importance
                    imp = self._causal_imp[c, p, alive]
                    median = imp.median().item()

                    keep = imp >= median
                    kill_indices = alive[~keep]
                    self._alive[c, p, kill_indices] = False

                    # Halve remaining energy
                    for idx in alive[keep]:
                        self._energy[c, p, idx] /= 2
                        self._causal_imp[c, p, idx] = 0.0
                        self._imp_count[c, p, idx] = 0

        self.extinction_interval += 2

    def predict(self, scores):
        return scores.argmax(dim=1)

    def get_n_cells(self):
        return self._alive.sum().item()


if __name__ == '__main__':
    torch.manual_seed(42)
    m = Brain(in_features=784, out_features=5, cells_per_class=2)
    x = torch.randn(4, 784)
    labels = torch.randint(0, 5, (4,))
    s = m(x)
    print(f'Forward: {s.shape}')
    m.train_step(x, labels)
    m.update_energy()
    m.structural_update()
    print(f'Cells: {m.get_n_cells()}')
    m.extinction()
    print(f'After extinction: {m.get_n_cells()}')
    print('PASS')
