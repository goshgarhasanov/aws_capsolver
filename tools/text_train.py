"""Train the BLS Poland CRNN+CTC reader.

Phase 1 (``--phase synth``): synthetic captchas generated on the fly (tools/text_synth.py).
Phase 2 (``--phase finetune``): continue from a checkpoint on a mix of synthetic images and
the augmented real TRAIN split (tools/text_data.py); the real TEST split is never touched.

Model selection uses only the fixed synthetic validation set (and the real train split
during fine-tuning). Evaluate on the real test split with tools/text_eval.py.

    .venv/bin/python tools/text_train.py --phase synth --steps 20000 --out data/text/ckpt/synth.pt
    .venv/bin/python tools/text_train.py --phase finetune --init data/text/ckpt/synth.pt \\
        --steps 3000 --out data/text/ckpt/ft.pt
    .venv/bin/python tools/text_train.py --export data/text/ckpt/ft.pt   # -> weights/bls_poland_ocr.pt
"""

from __future__ import annotations

import argparse
import io
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import text_data  # noqa: E402
import text_synth  # noqa: E402
from aws_capsolver.text.model import (ARCH, CRNN, constrained_decode, load_rgb,  # noqa: E402
                                      to_tensor)

WEIGHTS = ROOT / "weights" / "bls_poland_ocr.pt"


def encode(text: str) -> list[int]:
    return [int(c) + 1 for c in text]


def augment_real(jpeg: bytes, rng: random.Random) -> Image.Image:
    """Light, label-preserving augmentation of a real sample."""
    im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    dx, dy = rng.randint(-4, 4), rng.randint(-2, 2)
    canvas = Image.new("RGB", im.size, (255, 255, 255))
    canvas.paste(im, (dx, dy))
    im = canvas
    if rng.random() < 0.5:
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.8, 1.25))
    if rng.random() < 0.5:
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.9, 1.1))
    if rng.random() < 0.5:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=rng.choice((60, 75, 90)))
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return im


class Stream(IterableDataset):
    def __init__(self, seed: int, real: list[tuple[str, bytes, str]], real_frac: float):
        self.seed, self.real, self.real_frac = seed, real, real_frac

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(self.seed * 1000 + wid)
        while True:
            if self.real and rng.random() < self.real_frac:
                _, jpeg, text = rng.choice(self.real)
                im = augment_real(jpeg, rng)
            else:
                text = text_synth.random_text(rng)
                im = Image.open(io.BytesIO(text_synth.render(text, rng)))
            yield to_tensor(load_rgb(im)), torch.tensor(encode(text))


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


def synth_val(n: int = 2000, seed: int = 12345):
    rng = random.Random(seed)
    xs, texts = [], []
    for _ in range(n):
        t = text_synth.random_text(rng)
        jpeg = text_synth.render(t, rng, faithful=rng.random() < 0.5)
        xs.append(to_tensor(load_rgb(Image.open(io.BytesIO(jpeg)))))
        texts.append(t)
    return torch.stack(xs), texts


@torch.no_grad()
def accuracy(model, xs: torch.Tensor, texts: list[str]) -> float:
    model.eval()
    ok = 0
    for i in range(0, len(xs), 512):
        lp = model(xs[i:i + 512]).numpy()
        ok += sum(constrained_decode(l, 5)[0] == t for l, t in zip(lp, texts[i:i + 512]))
    model.train()
    return ok / len(texts)


def real_tensors(rows):
    return (torch.stack([to_tensor(load_rgb(Image.open(io.BytesIO(b)))) for _, b, _ in rows])
            if rows else None, [l for _, _, l in rows])


def train(args) -> None:
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    model = CRNN()
    if args.init:
        model.load_state_dict(torch.load(args.init, weights_only=True)["state_dict"])
    real_train = text_data.load_real("train") if args.phase == "finetune" else []
    ds = Stream(args.seed, real_train, args.real_frac)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate,
                    persistent_workers=True, prefetch_factor=4)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.1 if not args.init else 0.05)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    vx, vt = synth_val()
    rx, rt = real_tensors(real_train)
    best = -1.0
    t0 = time.time()
    model.train()
    for step, (x, y) in enumerate(dl, 1):
        lp = model(x)                                   # B, T, C
        T = lp.shape[1]
        loss = ctc(lp.permute(1, 0, 2), y, torch.full((len(x),), T, dtype=torch.long),
                   torch.full((len(x),), 5, dtype=torch.long))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            acc = accuracy(model, vx, vt)
            racc = accuracy(model, rx, rt) if rx is not None else float("nan")
            print(f"step {step} loss {loss.item():.4f} synth-val {acc:.4f} real-train {racc:.3f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
            if acc >= best:
                best = acc
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"arch": ARCH, "state_dict": model.state_dict(), "step": step,
                            "synth_val": acc}, args.out)
        if step >= args.steps:
            break
    print(f"best synth-val {best:.4f} -> {args.out}")


def export(ckpt: str) -> None:
    sd = torch.load(ckpt, weights_only=True)["state_dict"]
    sd = {k: v.to(torch.float32) for k, v in sd.items()}
    WEIGHTS.parent.mkdir(exist_ok=True)
    torch.save({"arch": ARCH, "length": 5, "alphabet": "0123456789", "state_dict": sd}, WEIGHTS)
    print(f"{WEIGHTS} {WEIGHTS.stat().st_size / 1e6:.2f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("synth", "finetune"), default="synth")
    ap.add_argument("--init")
    ap.add_argument("--out", default=str(ROOT / "data/text/ckpt/model.pt"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--real-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--export")
    args = ap.parse_args()
    if args.export:
        export(args.export)
    else:
        train(args)


if __name__ == "__main__":
    main()
