"""
DNA v3 — Overproduction + Energy Budget model.

Biology-inspired principles:
  1. Overproduction: each edge starts with many cells, competition decides who lives.
  2. Energy budget: cells have energy that depletes over time and is replenished by activity.
  3. Zero vitality thresholds — everything is governed by energy economics.
"""

import torch
import numpy as np


class Brain(torch.nn.Module):
    """
    DNA v3 — Overproduction + Energy Budget.

    Architecture:
      Weight: [in_features, cells_per_edge, out_features]
      Energy: [in_features, cells_per_edge]

    Per epoch:
      - Forward pass collects cell activations
      - Energy update: base_cost + activation_reward - inhibition_penalty
      - Cells with energy <= 0: marked dead, removed
      - Cells with energy > split_threshold: divide

    Forward: sum over cells, normalize, sigmoid.
    Training: standard BCE + backprop.
    """

    def __init__(self, in_features=784, out_features=10, initial_cells=10,
                 max_cells=32, energy_init=100, energy_split=150,
                 base_cost=1.0, activation_reward=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_cells = max_cells
        self.total_cell_slots = max_cells * 2  # fixed max allocation

        # Energy parameters
        self.energy_init = energy_init
        self.energy_split = energy_split
        self.base_cost = base_cost
        self.activation_reward = activation_reward

        # Weight: [in_features, cells, out_features]
        self.weight = torch.nn.Parameter(
            torch.zeros(in_features, self.total_cell_slots, out_features))

        # Energy: [in_features, cells]  — not a parameter, just state
        self.register_buffer('_energy', torch.full(
            (in_features, self.total_cell_slots), energy_init, dtype=torch.float))

        # Alive mask: [in_features, cells]
        self.register_buffer('_alive', torch.zeros(
            in_features, self.total_cell_slots, dtype=torch.bool))

        # Track how many cells per edge are actually in use
        self.active_count = torch.nn.Parameter(
            torch.full((in_features,), initial_cells, dtype=torch.long), requires_grad=False)

        # Activation buffer (accumulated per epoch)
        self.register_buffer('_activation_sum',
            torch.zeros(in_features, self.total_cell_slots))

        # Initialize first `initial_cells` per edge as alive
        self._alive[:, :initial_cells] = True

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.1)

    # ---- Forward ----

    def forward(self, x, act=torch.sigmoid):
        """
        x: [batch, in_features] or [batch, 1, in_features]
        Returns: [batch, out_features]
        """
        # Handle [batch, 1, in_features] from Flatten preserving channel dim
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        batch = x.shape[0]
        # Weighted sum over all cells: [batch, in_features, cells, out_features]
        # x: [batch, in_features, 1, 1]
        # weight: [1, in_features, cells, out_features]
        x_exp = x.unsqueeze(-1).unsqueeze(-1)  # [batch, in_features, 1, 1]
        w_exp = self.weight.unsqueeze(0)         # [1, in_features, cells, out_features]
        # Elementwise multiply then sum over input features
        # out: [batch, cells, out_features]
        # x: [batch, in_features], weight: [in_features, cells, out_features]
        out = torch.einsum('bi,icd->bcd', x, self.weight)

        # Only alive cells contribute
        alive_mask = self._alive.unsqueeze(0)    # [1, in_features, cells]
        # Sum over cells for each input feature, then average
        # Weighted by alive: [batch, cells, out_features] -> sum over cells -> [batch, out_features]
        # But we need per-edge handling... flatten instead:
        # After T steps: classification based on first-to-spike or most-spikes
        # Sum over cells dimension: [batch, out_features]
        out_sum = out.sum(dim=1)  # [batch, out_features]

        # Normalize by number of edges (not cells, keeps magnitude stable)
        return act(out_sum / self.in_features)

    # ---- Track Activations ----

    def track_activations(self, x):
        """Accumulate activations per cell for energy update."""
        with torch.no_grad():
            # Handle [batch, 1, in_features]
            if x.dim() == 3 and x.shape[1] == 1:
                x = x.squeeze(1)
            # Cell output: [batch, cells, out_features]
            cell_out = torch.einsum('bi,icd->bcd', x, self.weight).abs()
            # Mean absolute activation per cell, per sample
            act = cell_out.mean(dim=(0, 2))  # [cells] — average over batch & out_features
            # Expand to [in_features, cells]
            self._activation_sum += act.unsqueeze(0).expand(self.in_features, -1)

    # ---- Energy Update ----

    def update_energy(self):
        """Update energy for all cells based on epoch activity."""
        with torch.no_grad():
            # Reward: average activation × reward multiplier
            # _activation_sum is now sum of per-sample means
            reward = self._activation_sum * self.activation_reward

            # Cost: base cost for all alive cells (constant per epoch)
            cost = self.base_cost

            # Energy change
            energy_change = reward - cost
            self._energy += energy_change

            # Clamp: energy can't go negative or above split threshold * 2
            self._energy.clamp_(0, self.energy_split * 2)

            # Reset activation for next epoch
            self._activation_sum.zero_()

    # ---- Structural Update ----

    def structural_update(self):
        """
        Remove dead cells (energy == 0), split rich cells (energy > split_threshold).
        """
        with torch.no_grad():
            # ---- Kill: energy == 0 ----
            dead = (self._energy <= 0) & self._alive
            if dead.any():
                self._alive[dead] = False

            # ---- Split: energy > split_threshold AND below max_cells ----
            for feat in range(self.in_features):
                alive_indices = torch.where(self._alive[feat])[0]
                n_alive = len(alive_indices)

                if n_alive >= self.max_cells:
                    continue

                # Find cells above split threshold
                rich = torch.where((self._energy[feat] > self.energy_split) & self._alive[feat])[0]
                for idx in rich:
                    if n_alive >= self.max_cells:
                        break
                    # Split: find a dead slot
                    dead_slots = torch.where(~self._alive[feat])[0]
                    if len(dead_slots) == 0:
                        break  # no free slots
                    new_idx = dead_slots[0].item()

                    # Copy parent weight with perturbation
                    parent_w = self.weight[feat, idx, :].clone()
                    self.weight[feat, new_idx, :] = parent_w + torch.randn_like(parent_w) * 0.05
                    self._energy[feat, new_idx] = self.energy_init
                    self._alive[feat, new_idx] = True
                    # Parent loses some energy
                    self._energy[feat, idx] = self._energy[feat, idx] * 0.5

                    n_alive += 1

    def get_n_cells(self):
        return self._alive.sum().item()


if __name__ == '__main__':
    torch.manual_seed(42)
    m = Brain(in_features=28, out_features=5, initial_cells=10)
    print(f"Parameters: {sum(p.numel() for p in m.parameters())}")
    print(f"Alive cells: {m.get_n_cells()}")

    x = torch.rand(4, 28)
    y = m(x)
    print(f"Output shape: {y.shape}")

    # Simulate training step
    m.track_activations(x)
    m.update_energy()
    print(f"Energy range: [{m._energy.min().item():.1f}, {m._energy.max().item():.1f}]")
    print(f"Alive before structural update: {m.get_n_cells()}")
    m.structural_update()
    print(f"Alive after: {m.get_n_cells()}")
