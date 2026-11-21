"""Networks for GeeTest icon click (imported lazily; needs torch).

* ``IconDetector``: picture (3 x H x W, the "clean" half-resolution picture, see
  ``icon.clean_picture``) -> heat map (1 x H x W) of icon centres.
* ``IconEncoder``: a 40 x 40 crop around an icon centre (RGB) -> unit embedding.
* ``HintEncoder``: a 40 x 40 hint icon mask (1 channel, strokes = 1) -> unit embedding.
Crops and hints of the same icon get a high dot product (trained with InfoNCE).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

CROP = 40      # crop / hint side in clean-picture pixels
EMB = 128


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class IconDetector(nn.Module):
    """Small U-Net: centre heat-map logits at full input resolution."""

    def __init__(self, width: int = 24) -> None:
        super().__init__()
        w = width
        self.e1, self.e2, self.e3 = _block(3, w), _block(w, 2 * w), _block(2 * w, 4 * w)
        self.mid = _block(4 * w, 4 * w)
        self.d3, self.d2, self.d1 = _block(8 * w, 2 * w), _block(4 * w, w), _block(2 * w, w)
        self.out = nn.Conv2d(w, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        m = self.mid(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([F.interpolate(m, size=e3.shape[2:], mode="nearest"), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="nearest"), e1], 1))
        return self.out(d1)


class _Encoder(nn.Module):
    def __init__(self, cin: int, width: int = 32) -> None:
        super().__init__()
        w = width
        self.body = nn.Sequential(_block(cin, w), nn.MaxPool2d(2), _block(w, 2 * w), nn.MaxPool2d(2),
                                  _block(2 * w, 4 * w), nn.MaxPool2d(2), _block(4 * w, 4 * w))
        self.head = nn.Linear(4 * w, EMB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.body(x)
        f = torch.amax(f, dim=(2, 3))   # rotation / position tolerant pooling
        return F.normalize(self.head(f), dim=-1)


class IconEncoder(_Encoder):
    def __init__(self) -> None:
        super().__init__(3)


class HintEncoder(_Encoder):
    def __init__(self) -> None:
        super().__init__(1)


class IconModels(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detector = IconDetector()
        self.icon = IconEncoder()
        self.hint = HintEncoder()
        self.logit_scale = nn.Parameter(torch.tensor(2.6))
