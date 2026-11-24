"""Train the 4x4 dense cell head (aws_capsolver/recaptcha/dense.py) on cached features.

Inputs (tools/recaptcha_embed.py):
  --what bx4dense --backbone ViT-B-32:laion2b_s34b_b79k@512   patch tokens of the photo
  --what bx4      --backbone ViT-B-32:laion2b_s34b_b79k@224   16 cell crops + photo (--fusion)
Labels: bxding/recaptcha-67k 4x4 (SAM 3 masks: a cell is positive when the target
overlaps it). Per-class thresholds are tuned for challenge exact match on validation.

  .venv/bin/python tools/recaptcha_train_dense.py --fusion --out weights/recaptcha_dense.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from aws_capsolver.recaptcha.dense import build  # noqa: E402
from recaptcha_embed import out_path  # noqa: E402
from recaptcha_train import BX_MAP, answers_set, tune_threshold_challenges  # noqa: E402

DENSE = "ViT-B-32:laion2b_s34b_b79k@512"
CROPS = "ViT-B-32:laion2b_s34b_b79k@224"
MIN_CHALLENGES = 30


def load(split: str, fusion: bool):
    b = np.load(out_path(DENSE, "bx4dense", split))
    t = np.array([BX_MAP.get(str(x), "") for x in b["target"]])
    A = np.zeros((len(t), 16), bool)
    for i in range(len(t)):
        A[i, list(answers_set(b["answers"][i]))] = True
    C = None
    if fusion:
        c = np.load(out_path(CROPS, "bx4", split))
        assert list(c["ids"]) == list(b["ids"]), f"{split}: crop / dense rows differ"
        C = c["emb"]
    return b["emb"], C, A, t


def probs(net, E, C, kidx, batch=128):
    import torch

    out = []
    net.eval()
    with torch.inference_mode():
        for s in range(0, len(E), batch):
            x = torch.from_numpy(E[s:s + batch].astype(np.float32))
            c = torch.from_numpy(C[s:s + batch].astype(np.float32)) if C is not None else None
            out.append(torch.sigmoid(net(x, c)).numpy())
    P = np.concatenate(out)
    return P[np.arange(len(E)), :, kidx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", action="store_true", help="also use the 224 px cell crop embeddings")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--train", nargs="+", default=["train"])
    ap.add_argument("--out", default=str(ROOT / "weights" / "recaptcha_dense.pt"))
    a = ap.parse_args()
    import torch

    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    parts = [load(s, a.fusion) for s in a.train]
    E = np.concatenate([p[0] for p in parts])
    C = np.concatenate([p[1] for p in parts]) if a.fusion else None
    A = np.concatenate([p[2] for p in parts])
    t = np.concatenate([p[3] for p in parts])
    classes = sorted(c for c in set(t) if c and (t == c).sum() >= MIN_CHALLENGES)
    ci = {c: i for i, c in enumerate(classes)}
    keep = np.isin(t, classes)
    E, A, t = E[keep], A[keep], t[keep]
    C = C[keep] if C is not None else None
    print("train", len(E), {c: int((t == c).sum()) for c in classes}, flush=True)
    net = build(E.shape[2], len(classes), hidden=a.hidden, cell_dim=C.shape[2] if C is not None else 0)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-2)
    bs = 64
    steps = a.epochs * ((len(E) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
    kk = torch.tensor([ci[c] for c in t])
    Y = torch.from_numpy(A.astype(np.float32))
    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(E))
        tot = 0.0
        for s in range(0, len(E), bs):
            idx = perm[s:s + bs]
            ii = np.sort(idx.numpy())
            idx = torch.from_numpy(ii)
            x = torch.from_numpy(E[ii].astype(np.float32))
            c = torch.from_numpy(C[ii].astype(np.float32)) if C is not None else None
            L = net(x, c)[torch.arange(len(ii)), :, kk[idx]]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(L, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(ii)
        print(f"epoch {ep}: loss {tot / len(E):.4f} ({time.time() - t0:.0f}s)", flush=True)

    Ev, Cv, Av, tv = load("validation", a.fusion)
    k = np.isin(tv, classes)
    Ev, Av, tv = Ev[k], Av[k], tv[k]
    Cv = Cv[k] if Cv is not None else None
    Pv = probs(net, Ev, Cv, [ci[c] for c in tv])
    thr = {}
    for c in classes:
        s = tv == c
        if s.sum() >= 15:
            thr[c], em = tune_threshold_challenges(Pv[s], Av[s])
            print(f"  thr {c:14s} {thr[c]:.2f} val exact {em:.4f} (n={s.sum()}, @0.5 "
                  f"{((Pv[s] >= 0.5) == Av[s]).all(1).mean():.4f})")
        else:
            thr[c] = 0.5
    torch.save({"version": 1, "dense_backbone": DENSE, "crop_backbone": CROPS if a.fusion else None,
                "classes": [str(c) for c in classes], "thresholds": {str(k_): float(v) for k_, v in thr.items()},
                "hidden": a.hidden,
                "dim": int(E.shape[2]), "cell_dim": int(C.shape[2]) if C is not None else 0,
                "state": {k_: v.detach().clone() for k_, v in net.state_dict().items()},
                "trained_on": f"bxding/recaptcha-67k 4x4 {'+'.join(a.train)} ({len(E)} challenges); "
                              f"{a.epochs} epochs; {time.strftime('%Y-%m-%d')}"}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
