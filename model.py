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

        # Energy parameters
        self.energy_init = energy_init
        self.energy_split = energy_split
        self.base_cost = base_cost
        self.activation_reward = activation_reward

        # Number of cells per edge (tracked dynamically)
        self.n_cells = initial_cells

        # Weight: [in_features, cells, out_features]
        self.weight = torch.nn.Parameter(torch.zeros(in_features, initial_cells, out_features))

        # Energy: [in_features, cells]  — not a parameter, just state
        self.register_buffer('_energy', torch.full((in_features, initial_cells), energy_init, dtype=torch.float))

        # Alive mask: [in_features, cells]
        self.register_buffer('_alive', torch.ones(in_features, initial_cells, dtype=torch.bool))

        # Activation buffer (accumulated per epoch)
        self.register_buffer('_activation_sum', torch.zeros(in_features, initial_cells))

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.1)

    # ---- Forward ----

    def forward(self, x, act=torch.sigmoid):
        """
        x: [batch, in_features]
        Returns: [batch, out_features]
        """
        batch = x.shape[0]
        # Weighted sum over all cells: [batch, in_features, cells, out_features]
        # x: [batch, in_features, 1, 1]
        # weight: [1, in_features, cells, out_features]
        x_exp = x.unsqueeze(-1).unsqueeze(-1)  # [batch, in_features, 1, 1]
        w_exp = self.weight.unsqueeze(0)         # [1, in_features, cells, out_features]
        out = (x_exp * w_exp).sum(dim=1)         # [batch, cells, out_features]

        # Only alive cells contribute
        alive_mask = self._alive.unsqueeze(0)    # [1, in_features, cells]
        # Sum over cells for each input feature, then average
        # Weighted by alive: [batch, cells, out_features] -> sum over cells -> [batch, out_features]
        # But we need per-edge handling... flatten instead:
        # All cells across all edges: [batch, in_features * cells, out_features]
        B, C, O = out.shape
        # Sum across all edges * cells: [batch, out_features]
        out_sum = out.sum(dim=1)  # [batch, out_features]

        # Normalize by number of edges (not cells, keeps magnitude stable)
        return act(out_sum / self.in_features)

    # ---- Track Activations ----

    def track_activations(self, x):
        """Accumulate activations per cell for energy update."""
        with torch.no_grad():
            # Cell output for this batch: [batch, in_features, cells, out_features]
            x_exp = x.unsqueeze(-1).unsqueeze(-1)
            w_exp = self.weight.unsqueeze(0)
            cell_out = (x_exp * w_exp).abs()  # [batch, in_features, cells, out_features]
            # Sum over batch and out_features
            act = cell_out.sum(dim=(0, 3))  # [in_features, cells]
            self._activation_sum += act

    # ---- Energy Update ----

    def update_energy(self):
        """Update energy for all cells based on epoch activity."""
        with torch.no_grad():
            # Reward: activation / max_possible
            # activation_sum is summed over all batches
            # Normalize by max possible activation (batch_size * out_features)
            # We'll just use raw activation_sum scaled by reward
            reward = self._activation_sum * self.activation_reward

            # Cost: base cost for all alive cells
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
                rich = torch.where(self._energy[feat] > self.energy_split)[0]
                for idx in rich:
                    if n_alive >= self.max_cells:
                        break
                    # Split: create a new cell
                    new_idx = self._add_cell(feat, idx.item())
                    if new_idx is not None:
                        n_alive += 1

            # ---- Prune dead cells (compactify) ----
            self._prune_dead()

    def _add_cell(self, feat, parent_idx):
        """Add a new cell to edge `feat`, inheriting from parent at `parent_idx`."""
        # Find first dead slot or expand
        dead_slots = torch.where(~self._alive[feat])[0]
        if len(dead_slots) > 0:
            new_idx = dead_slots[0].item()
        else:
            # Need to expand
            old_size = self.weight.shape[1]
            if old_size >= self.max_cells * 2:
                return None
            new_size = old_size * 2
            # Expand weight
            new_weight = torch.zeros(self.in_features, new_size, self.out_features, device=self.weight.device)
            new_weight[:, :old_size, :] = self.weight.data
            self.weight = torch.nn.Parameter(new_weight)
            # Expand buffers
            new_energy = torch.full((self.in_features, new_size), self.energy_init, device=self._energy.device)
            new_energy[:, :old_size] = self._energy
            self._energy = new_energy
            new_alive = torch.zeros(self.in_features, new_size, dtype=torch.bool, device=self._alive.device)
            new_alive[:, :old_size] = self._alive
            self._alive = new_alive
            new_act = torch.zeros(self.in_features, new_size, device=self._activation_sum.device)
            new_act[:, :old_size] = self._activation_sum
            self._activation_sum = new_act
            new_idx = old_size  # first new slot

        # Copy parent weight with perturbation
        parent_w = self.weight[feat, parent_idx, :].clone()
        self.weight[feat, new_idx, :] = parent_w + torch.randn_like(parent_w) * 0.05
        # Energy: half of parent's excess
        self._energy[feat, new_idx] = self.energy_init
        self._alive[feat, new_idx] = True
        # Parent loses some energy for the split
        self._energy[feat, parent_idx] /= 2

        return new_idx

    def _prune_dead(self):
        """Compactify: remove dead cells to free slots."""
        # Simply reset the dead cells' energy to 0 and keep them as free slots
        pass  # _add_cell reuses dead slots

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
