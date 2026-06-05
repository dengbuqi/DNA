"""
DNA v13 — Full-image receptive fields with higher capacity.

Each cell sees ALL pixels (like v4), but with more cells per class.
Key: more capacity without patch partitioning.

Architecture: weight[out_features, slots, in_features=784]
"""

import torch
import torch.nn.functional as F


class Brain(torch.nn.Module):
    def __init__(self, in_features=784, out_features=10, cells_per_class=3,
                 max_cells_per_class=32, lr=0.01, base_cost=2.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.C = cells_per_class
        self.max_C = max_cells_per_class
        self.total_slots = max_cells_per_class * 2
        self.lr = lr
        self.base_cost = base_cost

        self.weight = torch.nn.Parameter(
            torch.zeros(out_features, self.total_slots, in_features))
        self.register_buffer('_energy', torch.full(
            (out_features, self.total_slots), 100.0, dtype=torch.float))
        self.register_buffer('_alive', torch.zeros(
            (out_features, self.total_slots), dtype=torch.bool))
        self.register_buffer('_causal_imp', torch.zeros(
            (out_features, self.total_slots)))
        self.register_buffer('_imp_count', torch.zeros(
            (out_features, self.total_slots)))
        self.register_buffer('_grace', torch.zeros(
            (out_features, self.total_slots), dtype=torch.long))
        self.register_buffer('_eligibility', torch.zeros(
            (out_features, self.total_slots)))

        self._alive[:, :cells_per_class] = True
        self._init_weights()
        self.epoch = 0
        self.extinction_interval = 5

    def _init_weights(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.5 / (self.in_features ** 0.5))

    def _cell_output(self, x):
        p = x.unsqueeze(1).unsqueeze(2)
        w = self.weight.unsqueeze(0)
        raw = torch.sigmoid((p * w).sum(dim=-1))
        return raw * self._alive.unsqueeze(0).float()

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)
        elif x.dim() == 3 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)
        raw = self._cell_output(x)
        scores = raw.sum(dim=2) / self._alive.sum(dim=1).unsqueeze(0).clamp(min=1)
        return scores

    def phase1_update(self, x, labels):
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)
        elif x.dim() == 3 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)
        target = F.one_hot(labels, self.out_features).float()
        with torch.no_grad():
            output = torch.sigmoid(
                (x.unsqueeze(1).unsqueeze(2) * self.weight.unsqueeze(0)).sum(dim=-1)
            )
            error = target.unsqueeze(-1) - output
            dw = self.lr * x.unsqueeze(1).unsqueeze(2) * error.unsqueeze(-1)
            dw = dw * self._alive.unsqueeze(0).unsqueeze(-1).float()
            self.weight.data.add_(dw.mean(dim=0))
            self.weight.data.clamp_(-0.5, 0.5)

    def measure_all_causal(self):
        with torch.no_grad():
            alive = self._alive
            x_val = 1.0
            w_sum = self.weight.sum(dim=-1)
            out_normal = torch.sigmoid(x_val * w_sum)
            importance = (out_normal - 0.5).abs()
            for c in range(self.out_features):
                a = torch.where(alive[c])[0]
                for s in a:
                    si = s.item()
                    cnt = self._imp_count[c, si].item()
                    self._causal_imp[c, si] = (
                        self._causal_imp[c, si] * cnt + importance[c, si].item()
                    ) / (cnt + 1)
                    self._imp_count[c, si] += 1

    def update_energy(self, x, labels):
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)
        elif x.dim() == 3 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)
        target = F.one_hot(labels, self.out_features).float()
        with torch.no_grad():
            output = torch.sigmoid(
                (x.unsqueeze(1).unsqueeze(2) * self.weight.unsqueeze(0)).sum(dim=-1)
            )
            closeness = 1.0 - (target.unsqueeze(-1) - output).abs()
            reward = closeness.mean(dim=0) * self._alive.float()
            self._energy += (reward - self.base_cost * self._alive.float())
            self._energy.clamp_(0, 300.0)

    def structural_update(self):
        with torch.no_grad():
            dead = (self._energy <= 0) & self._alive & (self._grace <= 0)
            self._alive[dead] = False
            for c in range(self.out_features):
                alive = torch.where(self._alive[c])[0]
                n = len(alive)
                if n >= self.max_C:
                    continue
                rich = torch.where((self._energy[c] > 150.0) & self._alive[c] & (self._grace[c] <= 0))[0]
                for idx in rich:
                    if n >= self.max_C:
                        break
                    dead_slots = torch.where(~self._alive[c])[0]
                    if len(dead_slots) == 0:
                        break
                    ni = dead_slots[0].item()
                    self.weight.data[c, ni] = self.weight[c, idx].clone() + torch.randn_like(self.weight[c, idx]) * 0.05
                    self._energy[c, ni] = 100.0
                    self._alive[c, ni] = True
                    self._grace[c, ni] = 5
                    self._energy[c, idx] *= 0.5
                    n += 1
            self._grace = (self._grace - 1).clamp(min=0)

    def extinction(self):
        print(f'[Extinction] epoch {self.epoch} | cells={self.get_n_cells()}')
        with torch.no_grad():
            for c in range(self.out_features):
                alive = torch.where(self._alive[c])[0]
                if len(alive) == 0:
                    for _ in range(min(3, self.max_C)):
                        dead_slots = torch.where(~self._alive[c])[0]
                        if len(dead_slots) > 0:
                            ni = dead_slots[0].item()
                            self._alive[c, ni] = True
                            self._grace[c, ni] = 5
                            self._energy[c, ni] = 100.0
                            self.weight.data[c, ni] = torch.randn_like(self.weight[c, ni]) * 0.5 / (self.in_features ** 0.5)
                    continue
                imp = self._causal_imp[c, alive]
                median = imp.median().item()
                keep = imp >= median
                kill_indices = alive[~keep]
                self._alive[c, kill_indices] = False
                for idx in alive[keep]:
                    self._energy[c, idx] /= 2
                    # Don't reset causal_imp — keep running average for next extinction
        self.extinction_interval += 2

    def predict(self, scores):
        return scores.argmax(dim=1)

    def get_n_cells(self):
        return self._alive.sum().item()
