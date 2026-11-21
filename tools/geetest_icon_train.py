#!/usr/bin/env python3
"""Train the GeeTest icon-click models on synthetic pictures.

Synthetic pictures imitate GeeTest v4 icon pictures at their "clean" half resolution
(150 x 100): a posterised photo, 3-6 icons from GeeTest's own icon set (collected hint
icons, data/geetest/_static/static_resources/icon/**) drawn as coloured strokes with a
dark drop shadow, rotated and scaled; then the 2x2 LED texture + JPEG round trip of the
real pictures, and back through ``icon.clean_picture`` exactly like at inference.

Backgrounds come from other collected GeeTest pictures (v4 slide / nine / word, v3 slide
full backgrounds); real v4 icon pictures are never used for training (they are the test
set, see tools/geetest_bench.py).

    .venv/bin/python tools/geetest_icon_train.py --steps 6000
"""
from __future__ import annotations

import argparse
import glob
import io
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aws_capsolver.geetest import icon as I  # noqa: E402

log = logging.getLogger("geetest_icon_train")
H, W = 100, I.WORK_WIDTH


def icon_files() -> list[str]:
    return sorted(glob.glob(str(ROOT / "data/geetest/_static/static_resources/icon/*/*/*.png")))


def background_files() -> list[str]:
    pats = ["data/geetest/v4_slide/*/bg.png", "data/geetest/v4_nine/*/image.jpg",
            "data/geetest/v4_word/*/image.jpg", "data/geetest/v3_slide/*/fullbg.jpg"]
    out = []
    for p in pats:
        out += sorted(glob.glob(str(ROOT / p)))
    return out


class Synth:
    def __init__(self, seed: int) -> None:
        self.rnd = random.Random(seed)
        self.icons = icon_files()
        self.bgs = background_files()
        self._ink: dict[int, np.ndarray] = {}
        self._mask: dict[int, np.ndarray] = {}
        self._bg: dict[int, Image.Image] = {}

    def ink(self, i: int) -> np.ndarray:
        if i not in self._ink:
            im = Image.open(self.icons[i]).convert("RGBA")
            a = np.asarray(im, np.float32) / 255.0
            ink = a[:, :, 3] * np.clip(1.0 - a[:, :, :3].mean(2), 0, 1)
            ys, xs = np.where(ink > 0.2)
            self._ink[i] = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            self._mask[i] = I.hint_mask(im)
        return self._ink[i]

    def mask(self, i: int) -> np.ndarray:
        self.ink(i)
        return self._mask[i]

    def background(self) -> np.ndarray:
        r = self.rnd
        k = r.randrange(len(self.bgs))
        if k not in self._bg:
            im = Image.open(self.bgs[k]).convert("RGB")
            if "fullbg" in self.bgs[k]:
                im = __import__("aws_capsolver.geetest.slide", fromlist=["x"]).unscramble_v3(im)
            self._bg[k] = im
        im = self._bg[k]
        if "v4_nine" in self.bgs[k]:  # a 3x3 collage: take one or two cells' worth
            cw, ch = im.width // 3, im.height // 3
            c, rr = r.randrange(3), r.randrange(3)
            im = im.crop((c * cw, rr * ch, (c + 1) * cw, (rr + 1) * ch))
        w, h = im.size
        cw = r.uniform(0.5, 1.0) * w
        chh = min(h, cw * H / W)
        x0, y0 = r.uniform(0, w - cw), r.uniform(0, h - chh)
        im = im.crop((int(x0), int(y0), int(x0 + cw), int(y0 + chh))).resize((W, H), Image.BILINEAR)
        if r.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = im.quantize(colors=r.randrange(5, 16), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).convert("RGB")
        a = np.asarray(im, np.float32)
        a = a * r.uniform(0.6, 1.0) + r.uniform(0, 30)
        gray = a.mean(2, keepdims=True)
        a = gray + (a - gray) * r.uniform(0.5, 1.0)
        return np.clip(a, 0, 255)

    def render_icon(self, i: int, size: float, angle: float, thick: int) -> np.ndarray:
        ink = self.ink(i)
        h, w = ink.shape
        side = int(max(h, w) * 1.5)
        sq = np.zeros((side, side), np.float32)
        sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = ink
        m = Image.fromarray((sq * 255).astype(np.uint8))
        if thick:
            m = m.filter(ImageFilter.MaxFilter(2 * thick * 2 + 1))
        m = m.rotate(angle, resample=Image.BILINEAR)
        out = side * size / max(h, w)
        m = m.resize((max(6, int(round(out))), max(6, int(round(out)))), Image.BOX)
        return np.asarray(m, np.float32) / 255.0

    def sample(self, n_min: int = 3, n_max: int = 6):
        r = self.rnd
        a = self.background()
        n = r.randint(n_min, n_max)
        placed: list[tuple[float, float, float]] = []
        items = []
        ids = r.sample(range(len(self.icons)), n)
        for i in ids:
            size = r.uniform(20, 31)
            for _ in range(40):
                cx, cy = r.uniform(size * 0.45, W - size * 0.45), r.uniform(size * 0.45, H - size * 0.45)
                if all(math.hypot(cx - px, cy - py) > (size + ps) * 0.5 + r.uniform(-2, 4) for px, py, ps in placed):
                    break
            else:
                continue
            placed.append((cx, cy, size))
            m = self.render_icon(i, size, r.uniform(-60, 60), r.choice([0, 1, 1, 2]))
            ghost = r.random() < 0.12
            if ghost:
                color = np.array([235, 235, 240], np.float32) * r.uniform(0.85, 1.0)
                opacity = r.uniform(0.35, 0.6)
            else:
                hsv = Image.new("HSV", (1, 1), (r.randrange(256), r.randrange(90, 245), r.randrange(185, 256)))
                color = np.asarray(hsv.convert("RGB"), np.float32)[0, 0]
                opacity = r.uniform(0.75, 1.0)
            mh, mw = m.shape
            x0, y0 = int(round(cx - mw / 2)), int(round(cy - mh / 2))
            # drop shadow
            if r.random() < 0.85:
                dx, dy = r.choice([1, 2]), r.choice([1, 2])
                sm = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3)), np.float32) / 255
                self._blend(a, sm * r.uniform(0.4, 0.8), np.array([20, 24, 45], np.float32), x0 + dx, y0 + dy)
            self._blend(a, m * opacity, color, x0, y0)
            items.append((cx, cy, i))
        return a.astype(np.uint8), items

    @staticmethod
    def _blend(a: np.ndarray, alpha: np.ndarray, color: np.ndarray, x0: int, y0: int) -> None:
        h, w = alpha.shape
        ys, xs = max(0, -y0), max(0, -x0)
        ye, xe = min(h, a.shape[0] - y0), min(w, a.shape[1] - x0)
        if ye <= ys or xe <= xs:
            return
        sub = a[y0 + ys:y0 + ye, x0 + xs:x0 + xe]
        al = alpha[ys:ye, xs:xe, None]
        sub[:] = sub * (1 - al) + color * al

    def finish(self, clean: np.ndarray) -> np.ndarray:
        """Texture + JPEG round trip, then the inference-time cleaning."""
        r = self.rnd
        big = np.empty((H * 2, W * 2, 3), np.float32)
        big[:] = r.uniform(50, 64) + np.random.default_rng(r.randrange(1 << 30)).normal(0, 2, big.shape)
        if r.random() < 0.85:
            big[1::2, 1::2] = clean
        else:  # untextured variant (screenshots / other renderings)
            big = np.asarray(Image.fromarray(clean).resize((W * 2, H * 2), Image.NEAREST), np.float32)
        buf = io.BytesIO()
        Image.fromarray(np.clip(big, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=r.randrange(55, 92))
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        c, *_ = I.clean_picture(im)
        return c


def heatmap(items, sigma: float = 1.6) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    hm = np.zeros((H, W), np.float32)
    for cx, cy, _ in items:
        hm = np.maximum(hm, np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
        iy, ix = min(H - 1, max(0, int(round(cy)))), min(W - 1, max(0, int(round(cx))))
        hm[iy, ix] = 1.0  # exactly one positive pixel per icon (focal loss)
    return hm


def make_batch(syn: Synth, bs: int):
    imgs, hms, crops, ids = [], [], [], []
    for _ in range(bs):
        clean, items = syn.sample()
        c = syn.finish(clean)
        imgs.append(c)
        hms.append(heatmap(items))
        for cx, cy, i in items:
            jx, jy = syn.rnd.uniform(-1.5, 1.5), syn.rnd.uniform(-1.5, 1.5)
            crops.append(I.crop_at(c, cx + jx, cy + jy))
            ids.append(i)
    return (np.stack(imgs), np.stack(hms), np.stack(crops), np.array(ids))


try:
    from torch.utils.data import IterableDataset as _IterableDataset
except ImportError:  # pragma: no cover
    _IterableDataset = object


class SynthStream(_IterableDataset):
    """Endless synthetic batches; every DataLoader worker gets its own random stream."""

    def __init__(self, batch: int) -> None:
        self.batch = batch

    def __iter__(self):
        import torch

        info = torch.utils.data.get_worker_info()
        syn = Synth(1000 + (info.id if info else 0) * 7919 + int(time.time()) % 100000)
        while True:
            yield make_batch(syn, self.batch)


def focal_loss(logits, target):
    import torch

    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.99).float()
    neg = 1 - pos
    pos_loss = -torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = -torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * neg
    n = pos.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n


def train(args) -> None:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    from aws_capsolver.geetest.icon_model import IconModels

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    dl = DataLoader(SynthStream(args.batch), batch_size=None, num_workers=args.workers, prefetch_factor=4 if args.workers else None)
    model = IconModels()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    syn_masks = Synth(5)
    t0 = time.time()
    it = iter(dl)
    for step in range(1, args.steps + 1):
        imgs, hms, crops, ids = (torch.as_tensor(v) for v in next(it))
        ids = ids.numpy()
        x = imgs.float().permute(0, 3, 1, 2) / 255.0
        y = hms.float()[:, None]
        det = focal_loss(model.detector(x), y)
        uniq, inv = np.unique(ids, return_inverse=True)
        masks = torch.from_numpy(np.stack([syn_masks.mask(int(u)) for u in uniq]))[:, None]
        ce = model.icon(crops.float().permute(0, 3, 1, 2))
        he = model.hint(masks)
        logits = model.logit_scale.exp().clamp(max=100) * ce @ he.T
        tgt = torch.from_numpy(inv).long()
        match = F.cross_entropy(logits, tgt)
        # symmetric term over the hints
        match = match + F.cross_entropy(logits.T, torch.from_numpy(
            np.array([int(np.where(inv == k)[0][0]) for k in range(len(uniq))])).long())
        loss = det + match
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0:
            acc = (logits.argmax(1) == tgt).float().mean().item()
            log.info("step %d det %.3f match %.3f acc %.3f (%.0fs)", step, det.item(), match.item(), acc,
                     time.time() - t0)
        if step % 1000 == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "step": step, "version": 1}, args.out)
    log.info("saved %s", args.out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "weights" / "geetest_icon.pt"))
    ap.add_argument("--preview", type=int, default=0, help="only write N synthetic previews to the scratch dir")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.preview:
        syn = Synth(3)
        sheet = Image.new("RGB", (W * 2 * 2, H * 2 * ((args.preview + 1) // 2)))
        for k in range(args.preview):
            clean, items = syn.sample()
            c = syn.finish(clean)
            sheet.paste(Image.fromarray(c).resize((W * 2, H * 2), Image.NEAREST), (k % 2 * W * 2, k // 2 * H * 2))
        out = Path(args.out).with_suffix(".preview.png")
        sheet.save(out)
        print(out)
        return 0
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
