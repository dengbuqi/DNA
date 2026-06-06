"""
DNA v15 — Nonlinear cells + WTA + Hebbian gain/bias + more cells.

Pure local learning: every update rule is local to the cell.
No gradients, no backprop — pure Hebbian principles.

Key:
  1. output = sigmoid(gain * (w·x + bias))
  2. gain updated by Hebbian confidence rule:
     - if cell's prediction matches label more often → increase gain (sharper)
     - if cell often wrong → decrease gain (flatter, more cautious)
  3. bias updated by homeostatic rule:
     - if average activation too high → decrease bias
     - if too low → increase bias
  4. WTA: within each class, only top-k cells remain active
     (k anneals: many → few)
  5. energy + extinction unchanged from v13
"""

import torch
import torch.nn.functional as F


class Brain(torch.nn.Module):
    def __init__(self, in_features=784, out_features=10, cells_per_class=8,
                 max_cells_per_class=32, lr=0.01, base_cost=0.5,
                 wta_k=1, wta_anneal_start=150, wta_anneal_end=500,
                 gain_lr=0.002, bias_lr=0.002,
                 extinction_mode='topk', extinction_keep=3):
        """
        extinction_mode: 'median' (old) or 'topk' (keep top N cells per class)
        extinction_keep: how many cells to keep per class per extinction
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.C = cells_per_class
        self.max_C = max_cells_per_class
        self.total_slots = max_cells_per_class * 2
        self.lr = lr
        self.base_cost = base_cost
        self.gain_lr = gain_lr
        self.bias_lr = bias_lr
        self.extinction_mode = extinction_mode
        self.extinction_keep = extinction_keep
        self.wta_k = wta_k
        self.wta_anneal_start = wta_anneal_start
        self.wta_anneal_end = wta_anneal_end

        # Core weights
        self.weight = torch.nn.Parameter(
            torch.zeros(out_features, self.total_slots, in_features))
        # Nonlinear params (learnable via Hebbian rules)
        self.gain = torch.nn.Parameter(
            torch.ones(out_features, self.total_slots))
        self.bias = torch.nn.Parameter(
            torch.zeros(out_features, self.total_slots))

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
        # Running average activation (for homeostatic bias)
        self.register_buffer('_avg_act', torch.zeros(
            (out_features, self.total_slots)))
        # Running prediction accuracy per cell
        self.register_buffer('_cell_correct', torch.zeros(
            (out_features, self.total_slots)))
        self.register_buffer('_cell_total', torch.zeros(
            (out_features, self.total_slots)))

        self._alive[:, :cells_per_class] = True
        self._init_weights()
        self.epoch = 0
        self.extinction_interval = 10

    def _init_weights(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.5 / (self.in_features ** 0.5))

    def _get_wta_k(self, n_alive):
        """Return current top-k count for each class, with annealing."""
        if self.epoch < self.wta_anneal_start:
            return 999  # no competition during warmup
        if self.epoch >= self.wta_anneal_end:
            return min(self.wta_k, n_alive)
        progress = (self.epoch - self.wta_anneal_start) / \
                   max(1, self.wta_anneal_end - self.wta_anneal_start)
        # Smooth anneal from all alive → wta_k
        k = n_alive - (n_alive - self.wta_k) * (1.0 - (1.0 - progress) ** 2)
        return max(self.wta_k, int(round(k)))

    def _compute_output(self, x, return_logit=False):
        """Compute sigmoid(gain * (w·x + bias)). Internal helper."""
        p = x.unsqueeze(1).unsqueeze(2)
        w = self.weight.unsqueeze(0)
        raw_logit = (p * w).sum(dim=-1)
        g = self.gain.unsqueeze(0)
        b = self.bias.unsqueeze(0)
        out = torch.sigmoid(g * (raw_logit + b))
        out = out * self._alive.unsqueeze(0).float()
        if return_logit:
            return out, raw_logit
        return out

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)
        elif x.dim() == 3 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)

        raw = self._compute_output(x)  # [B, out, slots]

        # WTA: within each class, only top-k cells contribute to score
        k = self._get_wta_k(self.max_C)
        if k < 999 and raw.shape[-1] > k:
            masked = raw * self._alive.unsqueeze(0).float()
            max_act = masked.max(dim=0).values  # [out, slots]
            for c in range(self.out_features):
                alive = torch.where(self._alive[c])[0]
                if len(alive) > k:
                    vals = max_act[c, alive]
                    _, top_idx = vals.topk(k)
                    keep = alive[top_idx]
                    mask = torch.zeros(self.total_slots, dtype=torch.bool, device=raw.device)
                    mask[keep] = True
                    raw[:, c, ~mask] = 0.0

        scores = raw.sum(dim=2) / self._alive.sum(dim=1).unsqueeze(0).clamp(min=1)
        return scores

    # ==================== Phase 1: Hebbian updates ====================

    def phase1_update(self, x, labels):
        """
        Three local update rules, all cell-autonomous:

        1. Weight: Δw = lr * x * (target - output) * gain * sig'
           (standard Hebbian with error-modulated gain scaling)

        2. Gain: Δgain = gain_lr * (confidence - 0.5)
           confidence = running accuracy of this cell
           If cell is more right than wrong → gain↑ (sharper)
           If more wrong than right → gain↓ (smoother)

        3. Bias: Δbias = bias_lr * (target_activation - current_activation)
           Homeostatic: maintains average activation around a setpoint.
        """
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)
        elif x.dim() == 3 and x.shape[1] == 1:
            x = x.view(x.shape[0], -1)

        target = F.one_hot(labels, self.out_features).float()
        with torch.no_grad():
            # --- Forward pass ---
            p = x.unsqueeze(1).unsqueeze(2)
            w = self.weight.unsqueeze(0)
            raw_logit = (p * w).sum(dim=-1)  # [B, out, slots]
            g = self.gain.unsqueeze(0)
            b = self.bias.unsqueeze(0)
            nlin = g * (raw_logit + b)
            output = torch.sigmoid(nlin)
            output = output * self._alive.unsqueeze(0).float()
            sig_prime = output * (1.0 - output)

            error = target.unsqueeze(-1) - output  # [B, out, slots]

            # === 1. Weight update (Hebbian) ===
            # Δw = lr * x * error * gain * sigmoid'
            grad_factor = error * g * sig_prime
            dw = self.lr * x.unsqueeze(1).unsqueeze(2) * grad_factor.unsqueeze(-1)
            dw = dw * self._alive.unsqueeze(0).unsqueeze(-1).float()
            self.weight.data.add_(dw.mean(dim=0))
            self.weight.data.clamp_(-1.0, 1.0)

            # === 2. Gain update (Hebbian confidence rule) ===
            # For each cell: how often is it right?
            # We track running accuracy per cell and adjust gain accordingly
            batch_preds = (raw_logit > 0).float()  # [B, out, slots], binary decision
            label_exp = target.unsqueeze(-1).float()  # [B, out, 1]
            # A cell is "correct for its class" if label is correct AND it fires
            cell_correct_this = (batch_preds == label_exp).float()  # [B, out, slots]
            batch_correct_rate = cell_correct_this.mean(dim=0) * self._alive.float()

            # Update running stats
            decay = 0.9
            self._cell_correct = self._cell_correct * decay + batch_correct_rate * (1 - decay)
            self._cell_total = (self._cell_total * decay + (1 - decay)).clamp(min=1e-6)

            cell_accuracy = self._cell_correct / self._cell_total  # [out, slots]

            # Δgain = gain_lr * (accuracy - 0.5)  → goes up if >50% correct, down if <50%
            dgain = self.gain_lr * (cell_accuracy - 0.5) * self._alive.float()
            self.gain.data.add_(dgain)
            self.gain.data.clamp_(0.1, 5.0)

            # === 3. Bias update (Homeostatic rule) ===
            # Maintain average activation around a setpoint
            act_setpoint = 0.5  # cells should fire ~50% of the time
            batch_act = output.mean(dim=0)  # [out, slots]
            # Running avg
            self._avg_act = self._avg_act * decay + batch_act * (1 - decay)
            # Δbias = bias_lr * (setpoint - avg_activation)
            # If activation too high → decrease bias (harder to fire)
            # If too low → increase bias (easier to fire)
            dbias = self.bias_lr * (act_setpoint - self._avg_act) * self._alive.float()
            self.bias.data.add_(dbias)
            self.bias.data.clamp_(-3.0, 3.0)

    # ==================== Causal importance ====================

    def measure_all_causal(self):
        """How much does a cell's output change when its input is zeroed?"""
        with torch.no_grad():
            w_sum = self.weight.sum(dim=-1)
            out_normal = torch.sigmoid(self.gain * (w_sum + self.bias))
            out_masked = torch.sigmoid(self.gain * self.bias)
            importance = (out_normal - out_masked).abs()

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
            output = self._compute_output(x)
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
                    self.weight.data[c, ni] = self.weight[c, idx].clone() + torch.randn_like(self.weight[c, idx]) * 0.05
                    self.gain.data[c, ni] = self.gain[c, idx].clone() + torch.randn_like(self.gain[c, idx]) * 0.05
                    self.bias.data[c, ni] = self.bias[c, idx].clone() + torch.randn_like(self.bias[c, idx]) * 0.02
                    self._energy[c, ni] = 100.0
                    self._alive[c, ni] = True
                    self._grace[c, ni] = 5
                    # Inherit running stats
                    self._cell_correct[c, ni] = self._cell_correct[c, idx] * 0.5
                    self._cell_total[c, ni] = self._cell_total[c, idx] * 0.5
                    self._avg_act[c, ni] = self._avg_act[c, idx]
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
                            self.weight.data[c, ni] = torch.randn_like(self.weight[c, ni]) * 0.5 / (self.in_features ** 0.5)
                            self.gain.data[c, ni] = 1.0
                            self.bias.data[c, ni] = 0.0
                    continue

                if self.extinction_mode == 'topk':
                    # Keep top-k cells by causal importance, kill rest
                    imp = self._causal_imp[c, alive]
                    k = min(self.extinction_keep, len(alive))
                    topk_vals, topk_idx = imp.topk(k)
                    keep = alive[topk_idx]
                    kill_indices = alive[~torch.isin(alive, keep)]
                    self._alive[c, kill_indices] = False
                    for idx in keep:
                        self._energy[c, idx] /= 2
                        self._causal_imp[c, idx] = 0.0
                        self._imp_count[c, idx] = 0
                else:
                    # Original median mode
                    imp = self._causal_imp[c, alive]
                    median = imp.median().item()
                    keep = imp >= median
                    kill_indices = alive[~keep]
                    self._alive[c, kill_indices] = False
                    for idx in alive[keep]:
                        self._energy[c, idx] /= 2
                        self._causal_imp[c, idx] = 0.0
                        self._imp_count[c, idx] = 0
        self.extinction_interval += 2

    def predict(self, scores):
        return scores.argmax(dim=1)

    def get_n_cells(self):
        return self._alive.sum().item()
