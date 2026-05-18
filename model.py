"""
DNA v4 — Final version.
Pure local Hebbian learning, no backpropagation.
Single-pixel cells, each with class affiliation.

Best result: 70.74% on MNIST test set.
"""

import torch
import torch.nn.functional as F


class Brain(torch.nn.Module):
    def __init__(self, in_features=784, out_features=10, cells_per_class=3,
                 max_cells_per_class=16, lr=0.01,
                 energy_init=100, energy_split=150, base_cost=5.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.C = cells_per_class
        self.max_C = max_cells_per_class
        self.total_slots = max_cells_per_class * 2
        self.lr = lr

        self.energy_init = energy_init
        self.energy_split = energy_split
        self.base_cost = base_cost

        self.weight = torch.nn.Parameter(
            torch.zeros(out_features, in_features, self.total_slots))

        self.register_buffer('_energy', torch.full(
            (out_features, in_features, self.total_slots), energy_init, dtype=torch.float))
        self.register_buffer('_alive', torch.zeros(
            (out_features, in_features, self.total_slots), dtype=torch.bool))

        self._alive[:, :, :cells_per_class] = True
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.5)

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        raw = torch.sigmoid(
            x.unsqueeze(1).unsqueeze(-1) * self.weight.unsqueeze(0)
        )
        raw = raw * self._alive.unsqueeze(0).float()
        scores = raw.sum(dim=(2, 3)) / self._alive.sum(dim=(1, 2)).unsqueeze(0).clamp(min=1)
        return scores

    def local_update(self, x, labels):
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        target = F.one_hot(labels, self.out_features).float()

        with torch.no_grad():
            output = torch.sigmoid(
                x.unsqueeze(1).unsqueeze(-1) * self.weight.unsqueeze(0)
            )
            target_exp = target.unsqueeze(-1).unsqueeze(-1)
            error = target_exp - output
            dw = self.lr * x.unsqueeze(1).unsqueeze(-1) * error
            dw = dw * self._alive.unsqueeze(0).float()
            self.weight.data.add_(dw.mean(dim=0))
            self.weight.data.clamp_(-3.0, 3.0)

    def update_energy(self, x, labels):
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        target = F.one_hot(labels, self.out_features).float()
        with torch.no_grad():
            output = torch.sigmoid(
                x.unsqueeze(1).unsqueeze(-1) * self.weight.unsqueeze(0)
            )
            target_exp = target.unsqueeze(-1).unsqueeze(-1)
            closeness = 1.0 - (target_exp - output).abs()
            reward = closeness.mean(dim=0) * self._alive.float()
            cost = torch.full(self._energy.shape, self.base_cost, dtype=torch.float,
                            device=self._energy.device) * self._alive.float()
            self._energy += (reward - cost)
            self._energy.clamp_(0, self.energy_split * 2)

    def structural_update(self):
        with torch.no_grad():
            dead = (self._energy <= 0) & self._alive
            self._alive[dead] = False
            for c in range(self.out_features):
                for p in range(self.in_features):
                    alive = torch.where(self._alive[c, p])[0]
                    n = len(alive)
                    if n >= self.max_C:
                        continue
                    rich = torch.where(
                        (self._energy[c, p] > self.energy_split) & self._alive[c, p]
                    )[0]
                    for idx in rich:
                        if n >= self.max_C:
                            break
                        dead_slots = torch.where(~self._alive[c, p])[0]
                        if len(dead_slots) == 0:
                            break
                        ni = dead_slots[0].item()
                        pw = self.weight[c, p, idx].clone()
                        self.weight[c, p, ni] = pw + torch.randn_like(pw) * 0.05
                        self._energy[c, p, ni] = self.energy_init
                        self._alive[c, p, ni] = True
                        self._energy[c, p, idx] *= 0.5
                        n += 1

    def predict(self, scores):
        return scores.argmax(dim=1)

    def get_n_cells(self):
        return self._alive.sum().item()
