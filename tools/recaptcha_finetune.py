"""Partially fine-tune the CLIP image tower for reCAPTCHA 3x3 tiles / single tiles.

Frozen CLIP features cap 3x3 exact-match at ~80 % (reCAPTCHA now distorts many tiles with
posterize / stipple filters). Fine-tuning the last few transformer blocks fixes most of
that while the first blocks stay shared with the frozen backbone at inference time.

Stages (each resumable):
  cache   run blocks [0, K) once and store the (n, tokens, width) fp16 activations
          for the bx 3x3 tiles (same unique-tile order as tools/recaptcha_embed.py) and
          the Danny tiles: data/recaptcha/tok/<what>_<split>_K<K>.npy
  train   train blocks [K, end) + ln_post + proj + a linear multi-label head on the
          cached activations (masked BCE, labels as in tools/recaptcha_train.py) and
          save the tail to weights/recaptcha_tail.pt
  embed   write tuned embeddings for every cached split to
          data/recaptcha/emb/<tuned spec>/... so tools/recaptcha_train.py can train the
          final MLP head + thresholds on them

  .venv/bin/python tools/recaptcha_finetune.py cache --splits bx3:train bx3:validation ...
  .venv/bin/python tools/recaptcha_finetune.py train --epochs 3
  .venv/bin/python tools/recaptcha_finetune.py embed
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from aws_capsolver.recaptcha import features  # noqa: E402
from aws_capsolver.recaptcha.backbone import TAIL_PATH, get_encoder  # noqa: E402
from recaptcha_embed import DATA, _prefetch, out_path  # noqa: E402
from recaptcha_train import BX_MAP, DANNY_MAP, TILE_CLASSES, answers_set, may_cooccur  # noqa: E402

BASE = "ViT-B-32:laion2b_s34b_b79k@224"
TOK = DATA / "tok"


def tok_path(what, split, K):
    return TOK / f"{what}_{split}_K{K}.npy"


def _open(p):
    from PIL import Image

    with Image.open(p) as im:
        return im.convert("RGB")


def crops_for(what, split):
    """Crops in exactly the order tools/recaptcha_embed.py embedded them."""
    if what == "danny":
        base = DATA / "danny" / "dataset_cls_full_57k" / split
        classes = sorted(os.listdir(DATA / "danny" / "dataset_cls_full_57k" / "train"))
        files = [f for c in classes if (base / c).is_dir() for f in sorted((base / c).iterdir())]
        yield from _prefetch(_open, files)
        return
    d = DATA / "bx" / "data" / "3x3" / split
    rows = [json.loads(line) for line in open(d / "metadata.jsonl")]
    rows = [r for r in rows if (d / r["file_name"]).is_file()]
    seen = set()

    def load(r):
        return features.model_crops(_open(d / r["file_name"]), 3)

    for crops in _prefetch(load, rows):
        for t in crops:
            h = hashlib.md5(t.tobytes()).digest()
            if h not in seen:
                seen.add(h)
                yield t


def stem(visual, x, K):
    import torch

    with torch.inference_mode():
        h = visual._embeds(x)
        for blk in visual.transformer.resblocks[:K]:
            h = blk(h)
    return h


def cmd_cache(a):
    import torch

    torch.set_num_threads(a.threads)
    enc = get_encoder(BASE)
    for spec in a.splits:
        what, split = spec.split(":")
        n = len(np.load(out_path(BASE, what, split))["emb"])
        p = tok_path(what, split, a.K)
        if p.exists():
            print("skip", p)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + ".tmp.npy")
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, 50, 768))
        buf, o, t0 = [], 0, time.time()
        for c in crops_for(what, split):
            buf.append(c)
            if len(buf) == 128:
                mm[o:o + len(buf)] = stem(enc.visual, features.to_tensor(buf, 224), a.K).numpy()
                o += len(buf)
                buf = []
                if o % (128 * 40) == 0:
                    print(f"  {spec}: {o}/{n} {o / (time.time() - t0):.0f} img/s", flush=True)
        if buf:
            mm[o:o + len(buf)] = stem(enc.visual, features.to_tensor(buf, 224), a.K).numpy()
            o += len(buf)
        assert o == n, f"{spec}: {o} crops but {n} embeddings"
        mm.flush()
        del mm
        os.replace(tmp, p)
        print(f"wrote {p} in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ labels
def tile_labels(what, split):
    """Y, M (n, K) aligned with the cached rows of one split (see recaptcha_train.tile_dataset)."""
    K = len(TILE_CLASSES)
    ci = {c: i for i, c in enumerate(TILE_CLASSES)}
    b = np.load(out_path(BASE, what, split))
    if what == "bx3":
        n = len(b["emb"])
        pos = np.zeros((n, K)); neg = np.zeros((n, K))
        for r in range(len(b["target"])):
            c = BX_MAP.get(str(b["target"][r]))
            if c is None:
                continue
            ans, unc = answers_set(b["answers"][r]), answers_set(b["uncertain"][r])
            for j in range(9):
                if j not in unc:
                    (pos if j in ans else neg)[b["tile_idx"][r, j], ci[c]] += 1
        Y = (pos > neg).astype(np.float32)
        M = (np.abs(pos - neg) > 0).astype(np.float32)
        return Y, M
    names = [DANNY_MAP[str(c)] for c in b["classes"]]
    y = b["y"]
    Y = np.zeros((len(y), K), np.float32); M = np.zeros((len(y), K), np.float32)
    for n_, yi in enumerate(y):
        a_ = names[yi]
        for c, k in ci.items():
            if a_ is None:
                M[n_, k] = 1
            elif c == a_:
                Y[n_, k] = 1; M[n_, k] = 1
            elif not may_cooccur(a_, c):
                M[n_, k] = 1
    return Y, M


class Tail:
    """Trainable copy of blocks [K, end) + ln_post + proj (+ a linear head during training)."""

    def __init__(self, visual, K, n_classes):
        import torch

        self.K = K
        self.blocks = copy.deepcopy(visual.transformer.resblocks[K:])
        self.ln = copy.deepcopy(visual.ln_post)
        self.proj = torch.nn.Parameter(visual.proj.detach().clone())
        self.head = torch.nn.Linear(self.proj.shape[1], n_classes)

    def params(self):
        return list(self.blocks.parameters()) + list(self.ln.parameters()) + [self.proj]

    def embed(self, h):
        for blk in self.blocks:
            h = blk(h)
        f = self.ln(h[:, 0]) @ self.proj
        return f / f.norm(dim=-1, keepdim=True)

    def train(self, flag=True):
        self.blocks.train(flag); self.ln.train(flag); self.head.train(flag)

    def state(self):
        import torch

        sd = {f"blocks.{k}": v.detach().to(torch.float16) for k, v in self.blocks.state_dict().items()}
        sd.update({f"ln.{k}": v.detach().to(torch.float16) for k, v in self.ln.state_dict().items()})
        sd["proj"] = self.proj.detach().to(torch.float16)
        return sd


def cmd_train(a):
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    enc = get_encoder(BASE)
    parts = []
    for spec in a.train:
        what, split = spec.split(":")
        T = np.load(tok_path(what, split, a.K), mmap_mode="r")
        Y, M = tile_labels(what, split)
        keep = np.flatnonzero(M.sum(1) > 0)
        if what == "danny" and a.danny_cap:
            # keep every tile of the classes bx3 lacks, subsample the rest
            b = np.load(out_path(BASE, "danny", split))
            cls = np.array([DANNY_MAP[str(c)] for c in b["classes"]])[b["y"]]
            rare = ~np.isin(cls, ["bicycle", "bus", "car", "crosswalk", "fire_hydrant"])
            rng = np.random.default_rng(0)
            common = np.flatnonzero(~rare)
            keep = np.union1d(np.flatnonzero(rare), rng.choice(common, min(a.danny_cap, len(common)), replace=False))
        parts.append((T, Y, M, keep))
        print(f"train {spec}: {len(keep)} rows", flush=True)
    index = [(pi, int(i)) for pi, (_, _, _, keep) in enumerate(parts) for i in keep]
    if a.limit:
        index = [index[i] for i in np.random.default_rng(2).choice(len(index), a.limit, replace=False)]
    tail = Tail(enc.visual, a.K, len(TILE_CLASSES))
    opt = torch.optim.AdamW([{"params": tail.params(), "lr": a.lr},
                             {"params": tail.head.parameters(), "lr": a.head_lr}], weight_decay=0.05)
    steps = a.epochs * (len(index) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr], total_steps=steps, pct_start=0.05)
    # validation: bx3 validation challenges, exact match at a tuned threshold per class
    Tv = np.load(tok_path("bx3", "validation", a.K), mmap_mode="r")
    bv = np.load(out_path(BASE, "bx3", "validation"))
    rng = np.random.default_rng(1)
    step = 0
    for ep in range(a.epochs):
        tail.train(True)
        order = rng.permutation(len(index))
        t0 = time.time()
        run = 0.0
        for k in range(0, len(order) - a.batch + 1, a.batch):
            sel = [index[i] for i in order[k:k + a.batch]]
            sel.sort()
            x = torch.from_numpy(np.stack([parts[p][0][i] for p, i in sel]).astype(np.float32))
            y = torch.from_numpy(np.stack([parts[p][1][i] for p, i in sel]))
            m = torch.from_numpy(np.stack([parts[p][2][i] for p, i in sel]))
            logit = tail.head(tail.embed(x) * a.scale)
            loss = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run = 0.98 * run + 0.02 * float(loss) if step > 1 else float(loss)
            if step % 100 == 0:
                done = k + a.batch
                print(f"  ep {ep} {done}/{len(order)} loss {run:.4f} {done / (time.time() - t0):.0f} img/s",
                      flush=True)
        em = evaluate(tail, Tv, bv, a)
        print(f"epoch {ep}: val exact {em}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"base": BASE, "K": a.K, "state": tail.state(), "head_W": tail.head.weight.detach(),
                    "head_b": tail.head.bias.detach(), "scale": a.scale, "trained_on":
                    f"{' + '.join(a.train)}; {ep + 1} epochs; {time.strftime('%Y-%m-%d')}"}, a.out)
        print("saved", a.out, flush=True)


def tail_embed(tail, T, scale=1.0, batch=256):
    import torch

    out = []
    tail.train(False)
    with torch.inference_mode():
        for k in range(0, len(T), batch):
            out.append(tail.embed(torch.from_numpy(np.asarray(T[k:k + batch], dtype=np.float32))).numpy())
    return np.concatenate(out)


def evaluate(tail, Tv, bv, a):
    import torch

    E = tail_embed(tail, Tv)
    with torch.inference_mode():
        P = torch.sigmoid(tail.head(torch.from_numpy(E) * a.scale)).numpy()
    res = {}
    tot = n = 0
    for c in ("bicycle", "bus", "car", "crosswalk", "fire_hydrant"):
        k = TILE_CLASSES.index(c)
        rows = [r for r in range(len(bv["target"])) if BX_MAP.get(str(bv["target"][r])) == c]
        Pc = P[bv["tile_idx"][rows], k]
        A = np.zeros((len(rows), 9), bool)
        for i, r in enumerate(rows):
            A[i, list(answers_set(bv["answers"][r]))] = True
        best = max(((Pc >= t) == A).all(1).mean() for t in np.linspace(0.1, 0.9, 33))
        res[c] = round(float(best), 4)
        tot += best * len(rows); n += len(rows)
    res["ALL"] = round(float(tot / n), 4)
    return res


def cmd_embed(a):
    import torch

    torch.set_num_threads(a.threads)
    enc = get_encoder(BASE)
    ck = torch.load(TAIL_PATH, map_location="cpu", weights_only=True)
    tail = Tail(enc.visual, ck["K"], len(TILE_CLASSES))
    load_tail_state(tail, ck["state"])
    tuned = f"{BASE}+tail"
    for p in sorted(TOK.glob(f"*_K{ck['K']}.npy")):
        what, split = p.stem.rsplit("_K", 1)[0].split("_", 1)
        dst = out_path(tuned, what, split)
        if dst.exists() and not a.force:
            print("skip", dst)
            continue
        src = dict(np.load(out_path(BASE, what, split)))
        src["emb"] = tail_embed(tail, np.load(p, mmap_mode="r")).astype(np.float16)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, **src)
        print("wrote", dst, flush=True)


def load_tail_state(tail, state):
    import torch

    tail.blocks.load_state_dict({k[7:]: v.float() for k, v in state.items() if k.startswith("blocks.")})
    tail.ln.load_state_dict({k[3:]: v.float() for k, v in state.items() if k.startswith("ln.")})
    with torch.no_grad():
        tail.proj.copy_(state["proj"].float())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--splits", nargs="+", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", nargs="+", default=["bx3:train", "danny:train"])
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--head-lr", type=float, default=1e-3)
    t.add_argument("--scale", type=float, default=10.0)
    t.add_argument("--danny-cap", type=int, default=8000)
    t.add_argument("--limit", type=int, default=0, help="train on a random subset (smoke tests)")
    t.add_argument("--out", default=str(TAIL_PATH))
    e = sub.add_parser("embed")
    e.add_argument("--force", action="store_true")
    for p in (c, t, e):
        p.add_argument("--K", type=int, default=10)
        p.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    {"cache": cmd_cache, "train": cmd_train, "embed": cmd_embed}[a.cmd](a)


if __name__ == "__main__":
    main()
