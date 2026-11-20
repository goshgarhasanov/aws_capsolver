"""
Network, alphabet and preprocessing of the general text-captcha reader (see general.py).

The model is a CRNN trained with CTC:

    RGB 64x256 -> residual CNN (stride 8 horizontally, height folded into channels) -> 32 steps
               -> 2-layer BiLSTM -> per-step logits over blank + 62 characters

Everything that must be identical between training (tools/general_train.py) and inference
(general.py) lives here: the alphabet, the image preprocessing and the CTC decoding helpers.
Only torch, numpy and Pillow are needed.
"""

import io
import string
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Alphabet: index 0 is the CTC blank, 1..62 are the characters below.
CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
BLANK = 0
NUM_CLASSES = len(CHARSET) + 1
CHAR_TO_INDEX = {c: i + 1 for i, c in enumerate(CHARSET)}

IMG_H = 64
IMG_W = 256
MAX_LABEL_LEN = 12

# For case-insensitive decoding / training every upper-case class is folded onto its
# lower-case partner: FOLD_TARGET[i] is the folded class of class i (digits map to
# themselves), FOLDED_CHARSET the 36-character alphabet of folded classes 1..36.
FOLDED_CHARSET = string.digits + string.ascii_lowercase
FOLD_TARGET = np.array([0] + [FOLDED_CHARSET.index(c.lower()) + 1 for c in CHARSET],
                       dtype=np.int64)


def encode(label: str, fold_case: bool = False) -> List[int]:
    """Label -> class indices (folded alphabet when `fold_case`); KeyError on a bad char."""
    if fold_case:
        return [FOLDED_CHARSET.index(c.lower()) + 1 for c in label]
    return [CHAR_TO_INDEX[c] for c in label]


# preprocessing -----------------------------------------------------------------
def flatten_rgb(img):
    """Any PIL image -> RGB, with transparency composited onto white."""
    from PIL import Image
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


def to_array(img) -> np.ndarray:
    """PIL image -> float32 CHW array in [-1, 1] at the model's input size."""
    from PIL import Image
    img = flatten_rgb(img)
    if img.size != (IMG_W, IMG_H):
        img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return (arr.transpose(2, 0, 1) / 127.5) - 1.0


def decode_bytes(image: bytes):
    """Encoded image bytes -> loaded PIL image (raises the usual PIL errors)."""
    from PIL import Image
    with Image.open(io.BytesIO(image)) as img:
        img.load()
        return img.copy()


# network -----------------------------------------------------------------------
def build_model():
    """The CRNN (~2.7 M parameters, ~7 ms per image on one CPU core group)."""
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(cout)
            self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(cout)
            self.skip = None
            if stride != 1 or cin != cout:
                self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                          nn.BatchNorm2d(cout))
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            out = self.act(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.act(out + (x if self.skip is None else self.skip(x)))

    class CRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 24, 3, 2, 1, bias=False), nn.BatchNorm2d(24),
                nn.ReLU(inplace=True))                                   # 32 x 128
            self.body = nn.Sequential(
                Block(24, 48, 2),                                        # 16 x 64
                Block(48, 96, 2),                                        # 8 x 32
                Block(96, 160, (2, 1)),                                  # 4 x 32
                Block(160, 160, 1),
            )
            self.drop = nn.Dropout(0.1)
            self.rnn = nn.LSTM(160 * 4, 192, num_layers=2, bidirectional=True,
                               batch_first=True, dropout=0.1)
            self.head = nn.Linear(384, NUM_CLASSES)

        def forward(self, x):                      # x: (B, 3, 64, 256)
            f = self.body(self.stem(x))            # (B, 160, 4, 32)
            b, ch, h, w = f.shape
            f = f.permute(0, 3, 1, 2).reshape(b, w, ch * h)
            f, _ = self.rnn(self.drop(f))
            return self.head(f)                    # (B, T=32, NUM_CLASSES) logits

    return CRNN()


# decoding ----------------------------------------------------------------------
def charset_mask(charset: Optional[str]) -> np.ndarray:
    """
    Boolean mask over NUM_CLASSES for a charset restriction; blank is always allowed.

    `charset` is None (everything), "digits", "letters", "alnum", "lower", "upper" or an
    explicit string of allowed characters (characters outside the alphabet are ignored).
    """
    mask = np.zeros(NUM_CLASSES, dtype=bool)
    mask[BLANK] = True
    if charset is None or charset == "alnum":
        allowed = CHARSET
    elif charset == "digits":
        allowed = string.digits
    elif charset == "letters":
        allowed = string.ascii_letters
    elif charset == "lower":
        allowed = string.ascii_lowercase
    elif charset == "upper":
        allowed = string.ascii_uppercase
    else:
        allowed = charset
    for ch in allowed:
        idx = CHAR_TO_INDEX.get(ch)
        if idx is not None:
            mask[idx] = True
    return mask


def log_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    z = logits - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def fold_case(logp: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Case-insensitive view of per-step log-probs: a letter gets the probability mass of
    both its cases. The merged value lives on the lower-case class when that one is
    allowed by `mask`, else on the upper-case class (so "upper"-only charsets still
    answer in upper case); the other class of the pair is masked out. Returns
    (logp', mask') over the same NUM_CLASSES classes.
    """
    out = np.where(mask[None, :], logp, -np.inf)
    out_mask = mask.copy()
    for ch in string.ascii_lowercase:
        lo, up = CHAR_TO_INDEX[ch], CHAR_TO_INDEX[ch.upper()]
        if not (mask[lo] or mask[up]):
            continue
        merged = np.logaddexp(logp[:, lo], logp[:, up])
        keep, drop = (lo, up) if mask[lo] else (up, lo)
        out[:, keep] = merged
        out[:, drop] = -np.inf
        out_mask[keep], out_mask[drop] = True, False
    return out, out_mask


def greedy_decode(logp: np.ndarray, mask: np.ndarray) -> Tuple[List[int], List[float]]:
    """Best-path CTC decode of (T, C) log-probs restricted to `mask`; (classes, probs)."""
    masked = np.where(mask[None, :], logp, -np.inf)
    best = masked.argmax(axis=1)
    out: List[int] = []
    probs: List[float] = []
    prev = BLANK
    for t, idx in enumerate(best.tolist()):
        p = float(np.exp(masked[t, idx]))
        if idx != BLANK and idx != prev:
            out.append(idx)
            probs.append(p)
        elif idx != BLANK and idx == prev:
            probs[-1] = max(probs[-1], p)
        prev = idx
    return out, probs


def ctc_beam_search(logp: np.ndarray, mask: np.ndarray, beam: int = 8,
                    max_len: Optional[int] = None) -> List[Tuple[Tuple[int, ...], float]]:
    """
    Prefix beam search over (T, C) log-probs restricted to `mask`.

    Returns up to `beam` (prefix, log-prob) pairs, best first. Prefixes longer than
    `max_len` are pruned during the search, so length limits shape the result instead
    of merely filtering it afterwards.
    """
    masked = np.where(mask[None, :], logp, -np.inf)
    allowed = np.flatnonzero(mask)
    allowed = allowed[allowed != BLANK]
    # beams: prefix -> (log p ending in blank, log p ending in non-blank)
    beams = {(): (0.0, -np.inf)}
    for t in range(masked.shape[0]):
        row = masked[t]
        # only extend with the classes that matter at this step
        cand = allowed[row[allowed] > row[allowed].max() - 12.0] if allowed.size else allowed
        nxt = {}

        def add(prefix, pb, pnb):
            ob, onb = nxt.get(prefix, (-np.inf, -np.inf))
            nxt[prefix] = (np.logaddexp(ob, pb), np.logaddexp(onb, pnb))

        for prefix, (pb, pnb) in beams.items():
            total = np.logaddexp(pb, pnb)
            add(prefix, total + row[BLANK], -np.inf)
            if prefix:
                last = prefix[-1]
                add(prefix, -np.inf, pnb + row[last])     # repeat collapses
            for c in cand.tolist():
                if max_len is not None and len(prefix) >= max_len:
                    break
                p = row[c]
                if prefix and prefix[-1] == c:
                    add(prefix + (c,), -np.inf, pb + p)   # needs a blank in between
                else:
                    add(prefix + (c,), -np.inf, total + p)
        ranked = sorted(nxt.items(), key=lambda kv: -np.logaddexp(*kv[1]))
        beams = dict(ranked[:beam])
    out = [(prefix, float(np.logaddexp(*v))) for prefix, v in beams.items()]
    out.sort(key=lambda kv: -kv[1])
    return out


def classes_to_text(classes: Sequence[int]) -> str:
    return "".join(CHARSET[i - 1] for i in classes)
