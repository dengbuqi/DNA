"""
DNA v3 — Overproduction + Energy Budget + Lateral Inhibition.

Biology-inspired principles:
  1. Overproduction: each edge starts with many cells, competition decides who lives.
  2. Energy budget: cells have energy that depletes and is replenished by activity.
  3. Lateral inhibition: within an edge, cells compete via softmax — winner gets activation,
     losers are suppressed. This drives specialization toward different output classes.
"""

import torch
import torch.nn.functional as F


class Brain(torch.nn.Module):
    """
    DNA v3 + Lateral Inhibition.

    Key change: each cell within an edge competes for each output class.
    On a given (pixel, class) pair, only the strongest cell's signal passes through,
    others are attenuated by softmax competition.

    Architecture:
      Weight: [in_features, cells_per_edge, out_features]
      Energy: [in_features, cells_per_edge]

    Forward with inhibition:
      - For each (pixel, class), compute raw cell outputs
      - Apply softmax competition across cells within each edge
      - Only the winning cell's output contributes to the final sum
      - This drives cells to specialize: each cell "owns" certain classes
    """

    def __init__(self, in_features=784, out_features=10, initial_cells=10,
                 max_cells=32, energy_init=100, energy_split=150,
                 base_cost=1.0, activation_reward=1.0,
                 inhibition_strength=3.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_cells = max_cells
        self.total_cell_slots = max_cells * 2

        self.energy_init = energy_init
        self.energy_split = energy_split
        self.base_cost = base_cost
        self.activation_reward = activation_reward
        self.inhibition_strength = inhibition_strength  # softmax temperature (lower=sharper)

        # Weight: [in_features, cells, out_features]
        self.weight = torch.nn.Parameter(
            torch.zeros(in_features, self.total_cell_slots, out_features))

        self.register_buffer('_energy', torch.full(
            (in_features, self.total_cell_slots), energy_init, dtype=torch.float))

        self.register_buffer('_alive', torch.zeros(
            in_features, self.total_cell_slots, dtype=torch.bool))

        self.register_buffer('_activation_sum',
            torch.zeros(in_features, self.total_cell_slots))

        self._alive[:, :initial_cells] = True
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.1)

    # ---- Forward with Lateral Inhibition ----

    def forward(self, x, act=torch.sigmoid):
        """
        x: [batch, in_features]
        Returns: [batch, out_features]

        For each pixel, cells compete via softmax over the output dimension.
        Only the cell(s) with the strongest response to each class contribute.
        """
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        # Raw cell outputs: [batch, cells, out_features]
        cell_out = torch.einsum('bi,icd->bcd', x, self.weight)

        # Apply alive mask: dead cells have output 0
        alive_mask = self._alive.unsqueeze(0).float()  # [1, in_features, cells]
        # cell_out is [batch, cells, out_features] — but cells = in_features * total_cell_slots
        # Wait — this is wrong. cell_out dim 1 = cells (all cells across ALL edges),
        # but alive_mask dim 1 = in_features.
        # We need to keep the [in_features, cells] structure throughout.

        # Let's restructure: compute per-edge output, then apply inhibition per-edge.
        # Weight: [in_features, cells, out_features]
        # We want: for each edge (in_feat), for each batch, for each class:
        #   raw = x[b, feat] * weight[feat, cell, class]  [cells, out_features]
        #   competitive = softmax(raw * inhibition_strength, dim=0) * raw
        #   sum over cells -> [out_features]

        B = x.shape[0]
        edge_outputs = torch.zeros(B, self.out_features, device=x.device)

        for feat in range(self.in_features):
            # Get alive cells for this edge
            alive = self._alive[feat]  # [total_cell_slots]
            if not alive.any():
                continue

            # Raw contribution of each cell to each class: [batch, n_alive, out_features]
            feat_weight = self.weight[feat, alive, :]  # [n_alive, out_features]
            px_val = x[:, feat:feat+1]  # [batch, 1]
            raw = px_val.unsqueeze(-1) * feat_weight.unsqueeze(0)  # [batch, n_alive, out_features]

            # Lateral inhibition: softmax competition across cells (dim=1)
            # For each output class, cells compete — strongest gets amplified, others suppressed
            # Apply softmax with temperature
            competitive = F.softmax(raw * self.inhibition_strength, dim=1)
            # Winning cells' output passes through
            inhibited = competitive * raw  # [batch, n_alive, out_features]

            # Sum over cells to get edge contribution to each class
            edge_outputs += inhibited.sum(dim=1)  # [batch, out_features]

        return act(edge_outputs / self.in_features)

    # ---- Track Activations (with inhibition awareness) ----

    def track_activations(self, x):
        """Accumulate activations per cell. Only cells that 'won' get rewarded."""
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        with torch.no_grad():
            B = x.shape[0]
            batch_act = torch.zeros_like(self._activation_sum)

            for feat in range(self.in_features):
                alive = self._alive[feat]
                if not alive.any():
                    continue

                feat_weight = self.weight[feat, alive, :]  # [n_alive, out_features]
                px_val = x[:, feat:feat+1]
                raw = px_val.unsqueeze(-1) * feat_weight.unsqueeze(0)  # [B, n_alive, out_features]

                # Competition — same as forward
                competitive = F.softmax(raw * self.inhibition_strength, dim=1)
                inhibited = competitive * raw

                # Reward each cell by its inhibited output (winner gets full, losers get little)
                # Mean over batch and out_features
                cell_reward = inhibited.mean(dim=(0, 2))  # [n_alive]
                batch_act[feat, alive] = cell_reward

            self._activation_sum += batch_act

    # ---- Energy Update ----

    def update_energy(self):
        with torch.no_grad():
            reward = self._activation_sum * self.activation_reward
            cost = self.base_cost
            self._energy += (reward - cost)
            self._energy.clamp_(0, self.energy_split * 2)
            self._activation_sum.zero_()

    # ---- Structural Update ----

    def structural_update(self):
        with torch.no_grad():
            # Kill: energy <= 0
            dead = (self._energy <= 0) & self._alive
            if dead.any():
                self._alive[dead] = False

            # Split: energy > split_threshold
            for feat in range(self.in_features):
                alive_indices = torch.where(self._alive[feat])[0]
                n_alive = len(alive_indices)

                if n_alive >= self.max_cells:
                    continue

                rich = torch.where((self._energy[feat] > self.energy_split) & self._alive[feat])[0]
                for idx in rich:
                    if n_alive >= self.max_cells:
                        break
                    dead_slots = torch.where(~self._alive[feat])[0]
                    if len(dead_slots) == 0:
                        break
                    new_idx = dead_slots[0].item()

                    parent_w = self.weight[feat, idx, :].clone()
                    self.weight[feat, new_idx, :] = parent_w + torch.randn_like(parent_w) * 0.05
                    self._energy[feat, new_idx] = self.energy_init
                    self._alive[feat, new_idx] = True
                    self._energy[feat, idx] = self._energy[feat, idx] * 0.5
                    n_alive += 1

    def get_n_cells(self):
        return self._alive.sum().item()


if __name__ == '__main__':
    torch.manual_seed(42)
    m = Brain(in_features=28, out_features=5, initial_cells=10,
              inhibition_strength=3.0)
    print(f"Params: {sum(p.numel() for p in m.parameters())}")
    print(f"Alive: {m.get_n_cells()}")

    x = torch.rand(4, 28)
    y = m(x)
    print(f"Output: {y.shape}")

    m.track_activations(x)
    m.update_energy()
    print(f"Energy: [{m._energy.min().item():.1f}, {m._energy.max().item():.1f}]")
    m.structural_update()
    print(f"Alive after: {m.get_n_cells()}")
