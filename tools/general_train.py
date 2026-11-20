"""
Train the general text-captcha CRNN (aws_capsolver/text/general_model.py) on CPU.

Data per batch: a share of on-the-fly synthetic captchas (tools/general_synth.py) and real
samples from the train packs built by tools/general_data.py (never their frozen test
split; the last `--val-per-source` train samples of each pack are a validation set).
Sources whose labels do not follow the glyph case (fold_train) get a case-folded CTC loss.

Resumable: the full state is saved to --ckpt every --save-every steps and on Ctrl-C;
running the same command again continues. The EMA weights are what gets exported.

    .venv/bin/python tools/general_train.py --steps 30000 --ckpt data/text_general/ckpt/run1.pt
    .venv/bin/python tools/general_train.py --export weights/general_ocr.pt --ckpt ...
    # fine-tune from a previous run with more real data and a lower learning rate:
    .venv/bin/python tools/general_train.py --init data/text_general/ckpt/run1.pt \\
        --ckpt data/text_general/ckpt/ft.pt --steps 8000 --lr 3e-4 --synth 0.3
"""

import argparse
import io
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from aws_capsolver.text import general_model as gm  # noqa: E402

PACKS = ROOT / "data" / "text_general" / "packs"


# data --------------------------------------------------------------------------
class Pack:
    """A train pack, memory-mapped (shared page cache across DataLoader workers)."""

    def __init__(self, name: str, fold: bool, val_n: int):
        meta = json.loads((PACKS / name / "train.json").read_text())
        self.name, self.fold = name, fold
        self.labels = meta["labels"]
        self.offsets = np.asarray(meta["offsets"], dtype=np.int64)
        self.blob = np.memmap(PACKS / name / "train.bin", dtype=np.uint8, mode="r")
        n = len(self.labels)
        self.val_n = min(val_n, n // 5)
        self.n_train = n - self.val_n

    def get(self, i: int):
        from PIL import Image
        data = self.blob[self.offsets[i]:self.offsets[i + 1]].tobytes()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return gm.flatten_rgb(im), self.labels[i]


def augment_real(img, rng: random.Random):
    """Mild augmentation for real samples: crop / pad jitter, rotation, colour, inversion."""
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    w, h = img.size
    if rng.random() < 0.6:
        dx0, dx1 = (int(rng.uniform(-0.04, 0.04) * w) for _ in range(2))
        dy0, dy1 = (int(rng.uniform(-0.06, 0.06) * h) for _ in range(2))
        box = (dx0, dy0, w - dx1, h - dy1)
        if box[2] - box[0] > w // 2 and box[3] - box[1] > h // 2:
            fill = img.getpixel((0, 0))
            canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), fill)
            canvas.paste(img, (-box[0], -box[1]))
            img = canvas
    if rng.random() < 0.3:
        img = img.rotate(rng.uniform(-4, 4), resample=Image.BILINEAR, expand=False,
                         fillcolor=img.getpixel((0, 0)))
    if rng.random() < 0.4:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.6, 1.4))
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.15:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.0, 1.5))
    if rng.random() < 0.1:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.08:
        img = ImageOps.invert(img)
    if rng.random() < 0.15:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(30, 90))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


class Stream:
    """Endless mix of synthetic and real samples: (CHW array, target classes, folded?)."""

    def __init__(self, args, sources):
        self.args, self.sources = args, sources

    def __iter__(self):
        import torch
        args = self.args
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        seed = (args.seed * 1000003 + wid * 7919 + int(time.time() * 1000)) & 0xFFFFFFFF
        rng = random.Random(seed)
        from general_synth import Generator
        gen = Generator(rng.getrandbits(32))
        packs = [Pack(n, fold, args.val_per_source) for n, fold in self.sources]
        weights = [min(p.n_train, args.source_cap) ** args.source_power for p in packs]
        while True:
            if not packs or rng.random() < args.synth:
                img, label, _ = gen.sample()
                fold = False
                if rng.random() < 0.15:          # synthetic samples get real-style aug too
                    img = augment_real(img, rng)
            else:
                p = rng.choices(packs, weights=weights)[0]
                try:
                    img, label = p.get(rng.randrange(p.n_train))
                except Exception:
                    continue
                img = augment_real(img, rng)
                fold = p.fold
            if not label or len(label) > gm.MAX_LABEL_LEN:
                continue
            try:
                target = gm.encode(label, fold_case=fold)
            except (KeyError, ValueError):
                continue
            yield gm.to_array(img), target, fold


def make_dataset(args, sources):
    import torch

    class _Iterable(Stream, torch.utils.data.IterableDataset):
        pass

    return _Iterable(args, sources)


def collate(batch):
    import torch
    x = torch.from_numpy(np.stack([b[0] for b in batch]))
    targets = [b[1] for b in batch]
    fold = torch.tensor([b[2] for b in batch], dtype=torch.bool)
    return x, targets, fold


# loss --------------------------------------------------------------------------
def ctc_loss(logits, targets, fold, ctc):
    """Mean CTC over the batch; folded samples are scored on the case-merged alphabet."""
    import torch
    logp = logits.log_softmax(-1)                     # (B, T, C)
    b, t, _ = logp.shape
    losses = []
    for mask, folded in ((~fold, False), (fold, True)):
        idx = mask.nonzero().flatten()
        if idx.numel() == 0:
            continue
        lp = logp[idx]
        if folded:
            fold_idx = torch.as_tensor(gm.FOLD_TARGET, device=lp.device)
            probs = lp.exp()
            merged = torch.zeros(lp.shape[0], t, len(gm.FOLDED_CHARSET) + 1, dtype=lp.dtype)
            merged = merged.index_add(2, fold_idx, probs)
            lp = merged.clamp_min(1e-12).log()
        tg = [targets[i] for i in idx.tolist()]
        flat = torch.tensor([c for s in tg for c in s], dtype=torch.long)
        lens = torch.tensor([len(s) for s in tg], dtype=torch.long)
        in_lens = torch.full((len(tg),), t, dtype=torch.long)
        loss = ctc(lp.transpose(0, 1), flat, in_lens, lens)
        losses.append(loss * len(tg))
    return sum(losses) / b


# evaluation --------------------------------------------------------------------
def evaluate(model, sources, val_n, limit=None):
    """Exact-match accuracy (case-insensitive) per source on the held-back val slices."""
    import torch
    model.eval()
    out = {}
    mask = gm.charset_mask(None)
    with torch.no_grad():
        for name, _ in sources:
            p = Pack(name, False, val_n)
            idx = list(range(p.n_train, len(p.labels)))[:limit]
            if not idx:
                continue
            ok = 0
            for start in range(0, len(idx), 128):
                chunk = idx[start:start + 128]
                items = [p.get(i) for i in chunk]
                x = torch.from_numpy(np.stack([gm.to_array(im) for im, _ in items]))
                lp = model(x).log_softmax(-1).numpy()
                for k, (_, label) in enumerate(items):
                    folded, fmask = gm.fold_case(lp[k], mask)
                    cls, _ = gm.greedy_decode(folded, fmask)
                    ok += gm.classes_to_text(cls).lower() == label.lower()
            out[name] = ok / len(idx)
    model.train()
    return out


def eval_synth(model, n=512, seed=12345):
    import torch
    from general_synth import Generator
    gen = Generator(seed)
    model.eval()
    ok = 0
    mask = gm.charset_mask(None)
    with torch.no_grad():
        for start in range(0, n, 128):
            items = [gen.sample() for _ in range(min(128, n - start))]
            x = torch.from_numpy(np.stack([gm.to_array(im) for im, _, _ in items]))
            lp = model(x).log_softmax(-1).numpy()
            for k, (_, label, case) in enumerate(items):
                cls, _ = gm.greedy_decode(lp[k], mask)
                text = gm.classes_to_text(cls)
                ok += (text == label) if case else (text.lower() == label.lower())
    model.train()
    return ok / n


# main --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", default=str(ROOT / "data/text_general/ckpt/run.pt"))
    ap.add_argument("--init", help="start from this checkpoint's (EMA) weights")
    ap.add_argument("--export", help="write the EMA weights of --ckpt here and exit")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--synth", type=float, default=0.5, help="share of synthetic samples")
    ap.add_argument("--sources", default="auto", help="comma list of packs, or auto (all)")
    ap.add_argument("--exclude", default="", help="comma list of packs to leave out")
    ap.add_argument("--source-cap", type=int, default=30000)
    ap.add_argument("--source-power", type=float, default=0.5)
    ap.add_argument("--val-per-source", type=int, default=200)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(ROOT / "tools"))
    from general_data import SOURCES

    if args.export:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "general_ocr/1", "charset": gm.CHARSET,
                    "img": [gm.IMG_H, gm.IMG_W], "state_dict": state["ema"],
                    "step": state["step"], "args": state.get("args")}, args.export)
        print(f"exported step {state['step']} -> {args.export} "
              f"({os.path.getsize(args.export) / 1e6:.1f} MB)")
        return

    exclude = set(filter(None, args.exclude.split(",")))
    if args.sources == "auto":
        names = [n for n, s in SOURCES.items() if not s.eval_only and n not in exclude
                 and (PACKS / n / "train.json").exists()]
    else:
        names = [n for n in args.sources.split(",") if n]
    sources = [(n, SOURCES[n].fold_train if n in SOURCES else False) for n in names]
    print("sources:", ", ".join(f"{n}{'(fold)' if f else ''}" for n, f in sources), flush=True)

    model = gm.build_model()
    ema = gm.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    step = 0
    ckpt = Path(args.ckpt)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        print(f"resumed {ckpt} at step {step}", flush=True)
    else:
        if args.init:
            state = torch.load(args.init, map_location="cpu", weights_only=False)
            sd = state.get("ema") or state.get("state_dict")
            model.load_state_dict(sd)
            print(f"initialised from {args.init}", flush=True)
        ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        prog = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))

    def save():
        tmp = str(ckpt) + ".tmp"
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                    "opt": opt.state_dict(), "step": step, "args": vars(args)}, tmp)
        os.replace(tmp, ckpt)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *a: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *a: stop.update(flag=True))

    loader = torch.utils.data.DataLoader(make_dataset(args, sources), batch_size=args.batch,
                                         num_workers=args.workers, collate_fn=collate,
                                         prefetch_factor=4, persistent_workers=True,
                                         multiprocessing_context="fork")
    ctc = torch.nn.CTCLoss(blank=gm.BLANK, reduction="mean", zero_infinity=True)
    model.train()
    t0, seen, run_loss = time.time(), 0, None
    log_path = ckpt.with_suffix(".log")
    for x, targets, fold in loader:
        if step >= args.steps or stop["flag"]:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        logits = model(x)
        loss = ctc_loss(logits, targets, fold, ctc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            d = min(args.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        step += 1
        seen += x.shape[0]
        lv = loss.item()
        run_loss = lv if run_loss is None else 0.98 * run_loss + 0.02 * lv
        if step % 50 == 0:
            rate = seen / (time.time() - t0)
            msg = f"step {step} loss {run_loss:.4f} lr {lr_at(step):.2e} {rate:.0f} img/s"
            print(msg, flush=True)
        if step % args.save_every == 0:
            save()
        if step % args.eval_every == 0:
            res = evaluate(ema, sources, args.val_per_source)
            syn = eval_synth(ema)
            mean = sum(res.values()) / max(1, len(res))
            line = {"step": step, "synth": round(syn, 4), "val_mean": round(mean, 4),
                    **{k: round(v, 4) for k, v in res.items()}}
            print("EVAL", json.dumps(line), flush=True)
            with open(log_path, "a") as fh:
                fh.write(json.dumps(line) + "\n")
    save()
    print(f"saved {ckpt} at step {step}", flush=True)


if __name__ == "__main__":
    main()
