"""Small CRNN + CTC digit reader for the BLS Poland captcha (shared by training and inference).

Input: the 160x55 RGB captcha (other sizes are resized), scaled to 0..1 and centred.
Colour is kept on purpose: the noise lines are red and the digits black, which is the
easiest cue the network can use.

Output: per-column log-probabilities over 11 classes (0 = CTC blank, 1..10 = digits 0..9),
T = 40 time steps (one per 4 input columns; a digit is ~23 px = ~6 steps wide).

Decoding is a best-path Viterbi constrained to exactly ``length`` labels (the captcha
always has 5 digits), so the reader never returns 4 or 6 characters.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

IMG_W, IMG_H = 160, 55
NUM_CLASSES = 11  # blank + 10 digits
ALPHABET = "0123456789"
ARCH = "crnn_v1"


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CRNN(nn.Module):
    def __init__(self, width: int = 1):
        super().__init__()
        c1, c2, c3, c4 = 32 * width, 64 * width, 96 * width, 128 * width
        self.features = nn.Sequential(
            _block(3, c1), nn.MaxPool2d(2),                 # 27 x 80
            _block(c1, c2), nn.MaxPool2d(2),                # 13 x 40
            _block(c2, c3), _block(c3, c3), nn.MaxPool2d((2, 1)),  # 6 x 40
            _block(c3, c4), _block(c4, c4),
        )
        self.rnn = nn.GRU(c4 * 6, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 55, 160) -> log-probs (B, T, C)."""
        f = self.features(x)                     # B, C, 6, 40
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)
        y, _ = self.rnn(f)
        return self.fc(self.drop(y)).log_softmax(-1)


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 (already IMG_H x IMG_W) -> 3xHxW float tensor."""
    t = torch.from_numpy(np.array(rgb, dtype=np.uint8, copy=True)).permute(2, 0, 1).float()
    return t.div_(255.0).sub_(0.5).div_(0.5)


def constrained_decode(logp: np.ndarray, length: int) -> tuple[str, float, list[float]]:
    """Best CTC path with exactly ``length`` labels.

    logp: (T, C) log-probabilities (class 0 = blank).
    Returns (text, confidence, per-character probabilities); the confidence is the
    product of the per-character probabilities (each the peak posterior of its label).
    States are (n labels emitted, last symbol s), s = 0 for blank or the digit class.
    """
    T, C = logp.shape
    if T < length:
        raise ValueError("sequence shorter than the requested length")
    neg = -1e30
    L = length + 1
    score = np.full((L, C), neg)
    score[0, 0] = logp[0, 0]
    score[1, 1:] = logp[0, 1:]
    back_n = np.zeros((T, L, C), dtype=np.int8)
    back_s = np.zeros((T, L, C), dtype=np.int8)
    labels = np.arange(1, C)
    for t in range(1, T):
        new = np.full((L, C), neg)
        bn = np.zeros((L, C), dtype=np.int8)
        bs = np.zeros((L, C), dtype=np.int8)
        # blank: from any state with the same n
        arg = score.argmax(1)
        new[:, 0] = score[np.arange(L), arg] + logp[t, 0]
        bn[:, 0] = np.arange(L)
        bs[:, 0] = arg
        # stay on the same label
        stay = score[:, 1:]
        new[:, 1:] = stay
        bn[:, 1:] = np.arange(L)[:, None]
        bs[:, 1:] = labels[None, :]
        # emit a new label c from (n-1, s) with s != c
        prev = score[:-1]                                  # (L-1, C)
        order = np.argsort(-prev, axis=1)
        top1, top2 = order[:, 0], order[:, 1]
        k = np.where(top1[:, None] == labels[None, :], top2[:, None], top1[:, None])
        emit = np.take_along_axis(prev, k, 1)              # (L-1, C-1)
        better = emit > new[1:, 1:]
        new[1:, 1:] = np.where(better, emit, new[1:, 1:])
        bn[1:, 1:] = np.where(better, np.arange(L - 1)[:, None], bn[1:, 1:])
        bs[1:, 1:] = np.where(better, k, bs[1:, 1:])
        new[:, 1:] += logp[t, 1:]
        score, back_n[t], back_s[t] = new, bn, bs
    n, s = length, int(np.argmax(score[length]))
    path = [0] * T
    for t in range(T - 1, -1, -1):
        path[t] = s
        if t:
            n, s = int(back_n[t, n, s]), int(back_s[t, n, s])
    probs_t = np.exp(logp)
    text, probs, prev_s = [], [], 0
    for t, s in enumerate(path):
        if s and s != prev_s:
            text.append(ALPHABET[s - 1])
            probs.append(float(probs_t[t, s]))
        elif s:
            probs[-1] = max(probs[-1], float(probs_t[t, s]))
        prev_s = s
    return "".join(text), float(np.prod(probs)) if probs else 0.0, probs


def greedy_decode(logp: np.ndarray) -> str:
    """Plain CTC best-path decoding (no length constraint); used for diagnostics."""
    out, prev = [], 0
    for s in logp.argmax(-1).tolist():
        if s and s != prev:
            out.append(ALPHABET[s - 1])
        prev = s
    return "".join(out)


def load_rgb(image) -> np.ndarray:
    """PIL image -> IMG_H x IMG_W x 3 uint8 (resized with bilinear filtering if needed)."""
    from PIL import Image

    im = image.convert("RGB")
    if im.size != (IMG_W, IMG_H):
        im = im.resize((IMG_W, IMG_H), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)
