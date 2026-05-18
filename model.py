"""
DNA v8 — Temporal Difference dopamine learning.

Key insight from v7 failure:
  Global average dopamine (0.34) is too weak and uniform.
  All cells get the same small positive signal — no differentiation.

TD fix:
  dopamine = current_accuracy - expected_accuracy
  If accuracy = 71% and expected = 68% -> dopamine = +0.03 (good surprise)
  If accuracy = 68% and expected = 68% -> dopamine = 0 (expected)
  If accuracy = 65% and expected = 68% -> dopamine = -0.03 (bad surprise)

  This captures deviations from expectation, not absolute performance.
"""

import torch
import torch.nn.functional as F


class Brain(torch.nn.Module):
    def __init__(self, in_features=784, out_features=10, cells_per_class=3,
                 max_cells_per_class=16, lr=0.01, trace_decay=0.9):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.C = cells_per_class
        self.max_C = max_cells_per_class
        self.total_slots = max_cells_per_class * 2
        self.lr = lr
        self.trace_decay = trace_decay

        # TD learning state
        self.expected_accuracy = 0.10  # start at random guess (10%)
        self.td_alpha = 0.1  # learning rate for expected accuracy update

        self.weight = torch.nn.Parameter(
            torch.zeros(out_features, in_features, self.total_slots))
        self.register_buffer('_eligibility', torch.zeros(out_features, in_features, self.total_slots))
        self.register_buffer('_alive', torch.zeros(
            (out_features, in_features, self.total_slots), dtype=torch.bool))
        self._alive[:, :, :cells_per_class] = True
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.5)

    def _compute(self, x):
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        raw = torch.sigmoid(x.unsqueeze(1).unsqueeze(-1) * self.weight.unsqueeze(0))
        raw = raw * self._alive.unsqueeze(0).float()
        scores = raw.sum(dim=(2, 3)) / self._alive.sum(dim=(1, 2)).unsqueeze(0).clamp(min=1)
        return scores

    def forward(self, x):
        return self._compute(x)

    def get_cell_out(self, x):
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        raw = torch.sigmoid(x.unsqueeze(1).unsqueeze(-1) * self.weight.unsqueeze(0))
        raw = raw * self._alive.unsqueeze(0).float()
        return raw

    def update_eligibility(self, batch_cell_out):
        """batch_cell_out: [batch, ...]"""
        act = batch_cell_out.mean(dim=0)
        self._eligibility *= self.trace_decay
        self._eligibility += act * self._alive.float()

    def compute_td_error(self, current_accuracy):
        """
        TD error = actual - expected.
        Positive: better than expected (dopamine surge)
        Negative: worse than expected (dopamine dip)
        """
        td_error = current_accuracy - self.expected_accuracy

        # Update expectation
        self.expected_accuracy += self.td_alpha * td_error

        return td_error

    def apply_td_update(self, td_error):
        """
        Δw = lr * td_error * eligibility
        td_error is a scalar (same for all cells).
        """
        if abs(td_error) < 0.005:  # tiny TD error, skip
            return

        with torch.no_grad():
            delta = self.lr * td_error * self._eligibility * self._alive.float()
            self.weight.data.add_(delta * 1.0)  # scale=1.0 (no extra 0.05)
            self.weight.data.clamp_(-3.0, 3.0)
            self._eligibility *= 0.8  # partial reset

    def phase1_update(self, x, labels):
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        target = F.one_hot(labels, self.out_features).float()
        with torch.no_grad():
            output = torch.sigmoid(x.unsqueeze(1).unsqueeze(-1) * self.weight.unsqueeze(0))
            target_exp = target.unsqueeze(-1).unsqueeze(-1)
            error = target_exp - output
            dw = self.lr * x.unsqueeze(1).unsqueeze(-1) * error
            dw = dw * self._alive.unsqueeze(0).float()
            self.weight.data.add_(dw.mean(dim=0))
            self.weight.data.clamp_(-3.0, 3.0)

    def predict(self, scores):
        return scores.argmax(dim=1)

    def get_n_cells(self):
        return self._alive.sum().item()
