"""
DNA v17 — EML cells (exp(w₁·x) - ln(w₂·x)) + pure local Hebbian.

Each cell has TWO weight vectors w₁, w₂, producing output:
  output = sigmoid(exp(w₁·x) - ln(w₂·x))

The EML operator eml(x, y) = exp(x) - ln(y) can generate all elementary
functions through nesting. This gives each cell far more expressive power
than a linear sigmoid cell.

Update rules (pure local, no BP):
  Δw₁ = lr * error * sig' * exp(w₁·x) * x
  Δw₂ = lr * error * sig' * (-1/(w₂·x)) * x

All terms are locally computable by the cell.
"""

import torch
import torch.nn.functional as F


def eml(x, y):
    """EML(x, y) = exp(x) - ln(y), numerically stable."""
    return torch.exp(x) - torch.log(y.clamp(min=1e-10))


class Brain(torch.nn.Module):
    def __init__(self, in_features=784, out_features=10, cells_per_class=8,
                 max_cells_per_class=32, lr=0.005, base_cost=0.5,
                 extinction_mode='soft_energy', extinction_keep=12,
                 lr_decay=0.5, w1_scale=1.0, w2_scale=1.0):
        """
        w1_scale, w2_scale: lr multipliers for w1/w2 branches.
        EML's exp branch (w1) has steep gradients, ln branch (w2) has flat gradients.
        We can compensate by scaling their LRs independently.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.C = cells_per_class
        self.max_C = max_cells_per_class
        self.total_slots = max_cells_per_class * 2
        self.lr = lr
        self.base_cost = base_cost
        self.w1_scale = w1_scale
        self.w2_scale = w2_scale
        self.extinction_mode = extinction_mode
        self.extinction_keep = extinction_keep
        self.lr_decay = lr_decay

        # Two weight sets per cell
        self.w1 = torch.nn.Parameter(torch.zeros(out_features, self.total_slots, in_features))
        self.w2 = torch.nn.Parameter(torch.zeros(out_features, self.total_slots, in_features))
        # w2 bias: ensures w2·x + bias > 0 so ln branch is never negative
        self.w2_bias = torch.nn.Parameter(torch.ones(out_features, self.total_slots))

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

        self._alive[:, :cells_per_class] = True
        self._init_weights()
        self.epoch = 0
        self.extinction_interval = 10

    def _init_weights(self):
        # w1: mean=0
        torch.nn.init.normal_(self.w1, mean=0.0, std=0.5 / (self.in_features ** 0.5))
        # w2: mean=0 keeps diversity
        torch.nn.init.normal_(self.w2, mean=0.0, std=0.5 / (self.in_features ** 0.5))

    def _cell_output(self, x):
        """
        x: [B, in_features]
        Returns: [B, out_features, slots]
        """
        p = x.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, in_features]
        l1 = (p * self.w1.unsqueeze(0)).sum(dim=-1)
        l2 = (p * self.w2.unsqueeze(0)).sum(dim=-1)
        l2 = l2 + self.w2_bias.unsqueeze(0)  # ensure positive, [B, out, slots]
        raw = torch.sigmoid(eml(l1, l2))
        return raw * self._alive.unsqueeze(0).float()

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)
        elif x.dim() == 3 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)
        raw = self._cell_output(x)
        scores = raw.sum(dim=2) / self._alive.sum(dim=1).unsqueeze(0).clamp(min=1)
        return scores

    # ==================== Phase 1: EML Hebbian update ====================

    def phase1_update(self, x, labels):
        """
        Pure local update for EML cells.

        Δw₁ = lr * w1_scale * error * sig' * exp(l₁) * x
        Δw₂ = lr * w2_scale * error * sig' * (-1/l₂) * x

        Where:
          l₁ = w₁·x, l₂ = w₂·x
          error = target - output
          sig' = output * (1 - output)
        """
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)
        elif x.dim() == 3 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)

        target = F.one_hot(labels, self.out_features).float()
        with torch.no_grad():
            p = x.unsqueeze(1).unsqueeze(2)
            l1 = (p * self.w1.unsqueeze(0)).sum(dim=-1)
            l2 = (p * self.w2.unsqueeze(0)).sum(dim=-1)
            l2 = l2 + self.w2_bias.unsqueeze(0)  # ensure positive
            l2_safe = l2.clamp(min=1e-10)
            output = torch.sigmoid(eml(l1, l2_safe))
            output = output * self._alive.unsqueeze(0).float()
            error = target.unsqueeze(-1) - output
            sig_prime = output * (1.0 - output)

            # Δw₁ = lr * w1_scale * error * sig' * exp(l₁) * x
            exp_part = torch.exp(l1).clamp(max=50)
            grad1 = error * sig_prime * exp_part * self.w1_scale
            dw1 = self.lr * p * grad1.unsqueeze(-1)
            dw1 = dw1 * self._alive.unsqueeze(0).unsqueeze(-1).float()
            self.w1.data.add_(dw1.mean(dim=0))
            self.w1.data.clamp_(-1.0, 1.0)

            # Δw₂ = lr * w2_scale * error * sig' * (-1/l₂) * x
            inv_part = -1.0 / l2_safe
            grad2 = error * sig_prime * inv_part * self.w2_scale
            dw2 = self.lr * p * grad2.unsqueeze(-1)
            dw2 = dw2 * self._alive.unsqueeze(0).unsqueeze(-1).float()
            self.w2.data.add_(dw2.mean(dim=0))
            self.w2.data.clamp_(-3.0, 3.0)

            # Δw2_bias = lr * w2_scale * error * sig' * (-1/l₂) * 1
            # (same as Δw2 but without x — bias has gradient 1)
            dbias = (self.lr * self.w2_scale * (error * sig_prime * inv_part)).mean(dim=0)
            self.w2_bias.data.add_(dbias)
            self.w2_bias.data.clamp_(0.1, 10.0)

    # ==================== Causal importance ====================

    def measure_all_causal(self):
        """Causal importance based on EML output sensitivity."""
        with torch.no_grad():
            x_val = 1.0
            w1_sum = self.w1.sum(dim=-1)
            w2_sum = self.w2.sum(dim=-1)
            out_normal = torch.sigmoid(eml(x_val * w1_sum, x_val * w2_sum))
            out_masked = torch.tensor(0.5, device=self.w1.device)  # sigmoid(0)
            # Use a simpler proxy: output strength
            importance = out_normal.abs()  # [out, slots]

            for c in range(self.out_features):
                a = torch.where(self._alive[c])[0]
                for s in a:
                    si = s.item()
                    cnt = self._imp_count[c, si].item()
                    self._causal_imp[c, si] = (
                        self._causal_imp[c, si] * cnt + importance[c, si].item()
                    ) / (cnt + 1)
                    self._imp_count[c, si] += 1

    # ==================== Energy & structure ====================

    def update_energy(self, x, labels):
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)
        elif x.dim() == 3 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)
        target = F.one_hot(labels, self.out_features).float()
        with torch.no_grad():
            output = self._cell_output(x)
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
                rich = torch.where(
                    (self._energy[c] > 150.0) & self._alive[c] & (self._grace[c] <= 0)
                )[0]
                for idx in rich:
                    if n >= self.max_C:
                        break
                    dead_slots = torch.where(~self._alive[c])[0]
                    if len(dead_slots) == 0:
                        break
                    ni = dead_slots[0].item()
                    self.w1.data[c, ni] = self.w1[c, idx].clone() + torch.randn_like(self.w1[c, idx]) * 0.05
                    self.w2.data[c, ni] = self.w2[c, idx].clone() + torch.randn_like(self.w2[c, idx]) * 0.05
                    self._energy[c, ni] = 100.0
                    self._alive[c, ni] = True
                    self._grace[c, ni] = 5
                    self._energy[c, idx] *= 0.5
                    n += 1
            self._grace = (self._grace - 1).clamp(min=0)

    def extinction(self):
        print(f'[Extinction] epoch {self.epoch} | cells={self.get_n_cells()} mode={self.extinction_mode}')
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
                            scale = 0.5 / (self.in_features ** 0.5)
                            self.w1.data[c, ni] = torch.randn_like(self.w1[c, ni]) * scale
                            self.w2.data[c, ni] = torch.randn_like(self.w2[c, ni]) * scale
                    continue

                if self.extinction_mode == 'topk':
                    imp = self._causal_imp[c, alive]
                    k = min(self.extinction_keep, len(alive))
                    _, topk_idx = imp.topk(k)
                    keep = alive[topk_idx]
                    kill_indices = alive[~torch.isin(alive, keep)]
                    self._alive[c, kill_indices] = False
                    for idx in keep:
                        self._energy[c, idx] /= 2
                        self._causal_imp[c, idx] = 0.0
                        self._imp_count[c, idx] = 0
                elif self.extinction_mode == 'soft_energy':
                    imp = self._causal_imp[c, alive]
                    median = imp.median().item()
                    for idx in alive:
                        if self._causal_imp[c, idx].item() < median:
                            self._energy[c, idx] *= 0.3
                        else:
                            self._energy[c, idx] /= 2
                        self._causal_imp[c, idx] = 0.0
                else:
                    # median mode
                    imp = self._causal_imp[c, alive]
                    median = imp.median().item()
                    keep = imp >= median
                    kill_indices = alive[~keep]
                    self._alive[c, kill_indices] = False
                    for idx in alive[keep]:
                        self._energy[c, idx] /= 2
        self.extinction_interval += 2

    def predict(self, scores):
        return scores.argmax(dim=1)

    def get_n_cells(self):
        return self._alive.sum().item()
