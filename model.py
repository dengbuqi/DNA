"""
DNA v10 — Temporal encoding via multi-frame scanning.

A single 28x28 image is scanned as 4 frames (14x14 quadrants).
Each frame: the cell sees 196 pooled pixels (2x2 per patch).
4 time steps = complete coverage of 784 pixels.

Each time step uses v4's local Hebbian rule independently.
Final classification averages over the 4 frames.

This gives cells temporal context without increasing weights per cell.
"""

import torch
import torch.nn.functional as F


class Brain(torch.nn.Module):
    def __init__(self, grid_size=14, n_frames=4, out_features=10,
                 cells_per_class=3, max_cells_per_class=16, lr=0.01):
        super().__init__()
        self.grid_size = grid_size  # 14 (28/2)
        self.n_frames = n_frames    # 4 quadrants
        self.out_features = out_features
        self.C = cells_per_class
        self.max_C = max_cells_per_class
        self.total_slots = max_cells_per_class * 2
        self.lr = lr

        self.n_patches = grid_size * grid_size  # 196 per frame

        # Weight: [n_frames, out_features, patches, slots]
        # Each frame has its own set of weights
        self.weight = torch.nn.Parameter(
            torch.zeros(n_frames, out_features, self.n_patches, self.total_slots))

        self.register_buffer('_alive', torch.zeros(
            (n_frames, out_features, self.n_patches, self.total_slots), dtype=torch.bool))
        self._alive[:, :, :, :cells_per_class] = True
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.5)

    def _extract_frames(self, x):
        """
        x: [batch, 784]
        Split 28x28 into 4 quadrants of 14x14.
        Frames: top-left, top-right, bottom-left, bottom-right.

        Returns: [batch, n_frames, 196]
        """
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        B = x.shape[0]
        img = x.view(B, 1, 28, 28)

        frames = []
        for row in range(2):
            for col in range(2):
                r_s, r_e = row * 14, (row + 1) * 14
                c_s, c_e = col * 14, (col + 1) * 14
                quadrant = img[:, :, r_s:r_e, c_s:c_e]  # [B, 1, 14, 14]
                # Max pool 1x1 (identity) to 14x14, flatten: [B, 196]
                frames.append(quadrant.reshape(B, -1))

        return torch.stack(frames, dim=1)  # [B, 4, 196]

    def forward(self, x):
        """
        Process all 4 frames, average scores.
        Separable inference: cells see one frame at a time in principle,
        but here we batch all frames for speed.
        """
        frames = self._extract_frames(x)  # [B, 4, 196]
        B = x.shape[0]

        frame_scores = torch.zeros(B, self.out_features, device=x.device)

        for f in range(self.n_frames):
            frame_input = frames[:, f, :]  # [B, 196]
            raw = torch.sigmoid(
                frame_input.unsqueeze(1).unsqueeze(-1) * self.weight[f].unsqueeze(0)
            )  # [B, out, 196, slots]
            raw = raw * self._alive[f].unsqueeze(0).float()
            scores = raw.sum(dim=(2, 3)) / self._alive[f].sum(dim=(1, 2)).unsqueeze(0).clamp(min=1)
            frame_scores += scores

        return frame_scores / self.n_frames

    def local_update(self, x, labels):
        """Update each frame's weights independently."""
        frames = self._extract_frames(x)
        target = F.one_hot(labels, self.out_features).float()

        with torch.no_grad():
            for f in range(self.n_frames):
                frame_input = frames[:, f, :]
                output = torch.sigmoid(
                    frame_input.unsqueeze(1).unsqueeze(-1) * self.weight[f].unsqueeze(0)
                )
                target_exp = target.unsqueeze(-1).unsqueeze(-1)
                error = target_exp - output
                dw = self.lr * frame_input.unsqueeze(1).unsqueeze(-1) * error
                dw = dw * self._alive[f].unsqueeze(0).float()
                self.weight.data[f].add_(dw.mean(dim=0))
                self.weight.data.clamp_(-3.0, 3.0)

    def predict(self, scores):
        return scores.argmax(dim=1)

    def get_n_cells(self):
        return self._alive.sum().item()


if __name__ == '__main__':
    torch.manual_seed(42)
    m = Brain(grid_size=14, n_frames=4, out_features=5, cells_per_class=3)
    print(f'Frames: {m.n_frames}, Patches/frame: {m.n_patches}, Cells: {m.get_n_cells()}')
    x = torch.randn(4, 784)
    s = m(x)
    print(f'Forward: {s.shape}')
    labels = torch.randint(0, 5, (4,))
    m.local_update(x, labels)
    print('PASS')
