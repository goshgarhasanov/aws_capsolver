"""4x4 dense cell head: patch-level classifier pooled into cells (multiple-instance learning).

A 4x4 challenge is one photo; a cell must be selected when ANY part of the object
overlaps it. The whole photo is encoded once at 512 px by ViT-B-32 (16x16 patch tokens,
exactly 4x4 patches per cell). A small network scores every patch for every class (patch
features + global CLS feature, a 3x3 convolution for spatial context); a cell's score is
the smooth maximum (log-sum-exp) of its 16 patch scores.
"""
from __future__ import annotations

import numpy as np

GRID = 16  # patches per side at 512 px / 32 px patches
CELL = GRID // 4


def build(dim: int, n_classes: int, hidden: int = 256, cell_dim: int = 0):
    """``cell_dim`` > 0: the forward pass also takes per-cell crop embeddings (n, 17, cell_dim)
    (16 cells + whole photo, as for the crop-based features); each cell's crop embedding is
    broadcast to its 4x4 patches and the whole-photo one to every patch."""
    import torch.nn as nn

    class DenseCellHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell_dim = cell_dim
            self.inp = nn.Conv2d(2 * dim + 2 * cell_dim, hidden, 1)
            self.ctx = nn.Conv2d(hidden, hidden, 3, padding=1)
            self.ctx2 = nn.Conv2d(hidden, hidden, 3, padding=1)
            self.out = nn.Conv2d(hidden, n_classes, 1)
            self.drop = nn.Dropout(0.2)
            self.tau = 4.0

        def forward(self, tok, crops=None):  # tok: (n, 1 + GRID*GRID, dim) -> cell logits (n, 16, K)
            import torch

            n = tok.shape[0]
            cls, p = tok[:, :1], tok[:, 1:]
            x = torch.cat([p, cls.expand(-1, p.shape[1], -1)], dim=2)
            x = x.transpose(1, 2).reshape(n, -1, GRID, GRID)
            if self.cell_dim:
                c = crops[:, :16].transpose(1, 2).reshape(n, -1, 4, 4)
                c = c.repeat_interleave(CELL, dim=2).repeat_interleave(CELL, dim=3)
                g = crops[:, 16][:, :, None, None].expand(-1, -1, GRID, GRID)
                x = torch.cat([x, c, g], dim=1)
            h = torch.relu(self.inp(self.drop(x)))
            h = h + torch.relu(self.ctx(h))
            h = h + torch.relu(self.ctx2(h))
            z = self.out(self.drop(h))  # (n, K, 16, 16)
            K = z.shape[1]
            z = z.reshape(n, K, 4, CELL, 4, CELL).permute(0, 2, 4, 1, 3, 5).reshape(n, 16, K, CELL * CELL)
            return torch.logsumexp(z * self.tau, dim=3) / self.tau - np.log(CELL * CELL) / self.tau

    return DenseCellHead()
