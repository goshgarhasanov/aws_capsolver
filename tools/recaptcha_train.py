"""Train the reCAPTCHA heads on cached CLIP embeddings (see tools/recaptcha_embed.py).

Tile head (3x3 tiles + single tiles), multi-label MLP with masked BCE:
  * bxding/recaptcha-67k 3x3 train: the challenge target is positive on the answer tiles
    and negative elsewhere; tiles the labeller flagged uncertain are ignored.
  * DannyLuna 57k: a tile is positive for its folder class, negative for classes that
    rarely appear together with it (street-scene pairs stay unknown), "Other" is negative
    for everything.
Cell head (4x4): MLP over [cell, whole photo, neighbour mean] trained on bx 4x4 train,
then a tiny neighbour-context stage trained on out-of-fold first-stage logits.
Per-class thresholds are tuned on the validation splits (bx validation; Danny val for
classes absent from bx).

  .venv/bin/python tools/recaptcha_train.py --backbones ViT-B-32:laion2b_s34b_b79k@224 \
      --out weights/recaptcha_head.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aws_capsolver.recaptcha.model import cell_inputs, context_inputs  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from recaptcha_embed import out_path  # noqa: E402

DANNY_MAP = {"Bicycle": "bicycle", "Bridge": "bridge", "Bus": "bus", "Car": "car", "Chimney": "chimney",
             "Crosswalk": "crosswalk", "Hydrant": "fire_hydrant", "Motorcycle": "motorcycle",
             "Mountain": "mountain", "Other": None, "Palm": "palm_tree", "Stair": "stairs",
             "Tractor": "tractor", "Traffic Light": "traffic_light"}
BX_MAP = {"bicycle": "bicycle", "bus": "bus", "car": "car", "chimney": "chimney", "crosswalk": "crosswalk",
          "fire hydrant": "fire_hydrant", "motorcycle": "motorcycle", "parking meter": "parking_meter",
          "stairs": "stairs", "taxi": "taxi", "tractor": "tractor", "traffic light": "traffic_light"}
TILE_CLASSES = ["bicycle", "bridge", "bus", "car", "chimney", "crosswalk", "fire_hydrant", "motorcycle",
                "mountain", "palm_tree", "parking_meter", "stairs", "taxi", "tractor", "traffic_light"]
STREET = {"car", "bus", "motorcycle", "bicycle", "traffic_light", "crosswalk", "fire_hydrant", "parking_meter",
          "taxi", "tractor", "bridge"}
MIN_CELL_CHALLENGES = 60


def may_cooccur(a: str, b: str) -> bool:
    """Pairs whose Danny negatives are unreliable (both often visible in one tile)."""
    if a == b:
        return True
    if "car" in (a, b) or "taxi" in (a, b):
        return True
    if a in STREET and b in STREET:
        return True
    pair = {a, b}
    return pair in ({"palm_tree", "mountain"}, {"bridge", "mountain"}, {"stairs", "chimney"},
                    {"palm_tree", "traffic_light"}, {"palm_tree", "bus"}, {"mountain", "bridge"})


def load(specs: list[str], what: str, split: str) -> dict:
    parts = [dict(np.load(out_path(s, what, split), allow_pickle=False)) for s in specs]
    d = parts[0]
    d["emb"] = np.concatenate([p["emb"].astype(np.float32) for p in parts], axis=-1)
    return d


def answers_set(row) -> set[int]:
    return {int(v) for v in row if v >= 0}


# ------------------------------------------------------------------ datasets
def tile_dataset(specs, split_bx: str, split_danny: str | None, use_danny=True):
    """X (n, D), Y (n, K) in {0,1}, M (n, K) weights (0 = unknown)."""
    K = len(TILE_CLASSES)
    ci = {c: i for i, c in enumerate(TILE_CLASSES)}
    Xs, Ys, Ms = [], [], []
    if split_bx:
        b = load(specs, "bx3", split_bx)
        U = b["emb"]
        pos = np.zeros((len(U), K))
        neg = np.zeros((len(U), K))
        for r in range(len(b["target"])):
            c = BX_MAP.get(str(b["target"][r]))
            if c is None:
                continue
            ans = answers_set(b["answers"][r])
            unc = answers_set(b["uncertain"][r])
            for j in range(9):
                if j in unc:
                    continue
                u = b["tile_idx"][r, j]
                (pos if j in ans else neg)[u, ci[c]] += 1
        Y = (pos > neg).astype(np.float32)
        M = ((pos + neg) > 0).astype(np.float32) * (np.abs(pos - neg) > 0)  # conflicting votes -> unknown
        keep = M.sum(1) > 0
        Xs.append(U[keep]); Ys.append(Y[keep]); Ms.append(M[keep])
    if use_danny and split_danny:
        d = load(specs, "danny", split_danny)
        names = [DANNY_MAP[str(c)] for c in d["classes"]]
        Y = np.zeros((len(d["y"]), K), np.float32)
        M = np.zeros((len(d["y"]), K), np.float32)
        for n, yi in enumerate(d["y"]):
            a = names[yi]
            for c, k in ci.items():
                if a is None:
                    M[n, k] = 1.0
                elif c == a:
                    Y[n, k] = 1.0; M[n, k] = 1.0
                elif not may_cooccur(a, c):
                    M[n, k] = 1.0
        Xs.append(d["emb"]); Ys.append(Y); Ms.append(M)
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ms)


# ------------------------------------------------------------------ training
def train_mlp(X, Y, M, hidden=512, epochs=30, lr=1e-3, wd=1e-4, dropout=0.3, seed=0, batch=512, pos_weight=None):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    D, K = X.shape[1], Y.shape[1]
    layers = [nn.Dropout(0.1)]
    if hidden:
        layers += [nn.Linear(D, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, K)]
    else:
        layers += [nn.Linear(D, K)]
    net = nn.Sequential(*layers)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    Xt, Yt, Mt = (torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in (X, Y, M))
    n = len(Xt)
    steps = epochs * ((n + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    pw = torch.ones(K) if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            logit = net(Xt[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(logit, Yt[idx], weight=Mt[idx], pos_weight=pw,
                                                                  reduction="sum") / Mt[idx].sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
    net.eval()
    lin = [m for m in net if isinstance(m, nn.Linear)]
    return [(m.weight.detach().numpy().copy(), m.bias.detach().numpy().copy()) for m in lin]


def mlp(layers, x):
    for i, (W, b) in enumerate(layers):
        x = x @ W.T + b
        if i < len(layers) - 1:
            x = np.maximum(x, 0)
    return x


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


# ------------------------------------------------------------------ thresholds
def tune_threshold_challenges(P, A, grid=np.linspace(0.05, 0.95, 91)):
    """P (n, cells) probs, A (n, cells) bool truth -> threshold maximising exact match."""
    best = (-1, 0.5)
    for t in grid:
        em = ((P >= t) == A).all(1).mean()
        best = max(best, (em + 1e-9 * -abs(t - 0.5), t))
    return float(best[1]), float(best[0])


def tune_threshold_tiles(p, y):
    best = (-1, 0.5)
    for t in np.linspace(0.05, 0.95, 91):
        acc = ((p >= t) == y).mean()
        best = max(best, (acc - 1e-9 * abs(t - 0.5), t))
    return float(best[1]), float(best[0])


def bx3_challenges(specs, split):
    b = load(specs, "bx3", split)
    X = b["emb"][b["tile_idx"]]  # (n, 9, D)
    A = np.zeros((len(X), 9), bool)
    for r in range(len(X)):
        for j in answers_set(b["answers"][r]):
            A[r, j] = True
    tg = np.array([BX_MAP.get(str(t), "") for t in b["target"]])
    return X, A, tg, b


def bx4_challenges(specs, split):
    b = load(specs, "bx4", split)
    A = np.zeros((len(b["emb"]), 16), bool)
    for r in range(len(A)):
        for j in answers_set(b["answers"][r]):
            A[r, j] = True
    tg = np.array([BX_MAP.get(str(t), "") for t in b["target"]])
    return b["emb"], A, tg, b


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", required=True, help="tile head features")
    ap.add_argument("--cell-backbones", nargs="+", default=None, help="4x4 cell head features (default: same)")
    ap.add_argument("--out", default=str(ROOT / "weights" / "recaptcha_head.pt"))
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--cell-hidden", type=int, default=512)
    ap.add_argument("--cell-epochs", type=int, default=30)
    ap.add_argument("--no-danny", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tile-only", action="store_true", help="skip the 4x4 cell head")
    a = ap.parse_args()
    import torch

    torch.set_num_threads(a.threads)
    specs = a.backbones
    t0 = time.time()

    # ---- tile head
    X, Y, M = tile_dataset(specs, "train", "train", use_danny=not a.no_danny)
    print(f"tile train: {len(X)} rows, labelled pos/neg per class:")
    for k, c in enumerate(TILE_CLASSES):
        print(f"  {c:14s} pos {int((Y[:, k] * M[:, k]).sum()):6d} neg {int(((1 - Y[:, k]) * M[:, k]).sum()):6d}")
    tile = train_mlp(X, Y, M, hidden=a.hidden, epochs=a.epochs)
    print(f"tile head trained in {time.time() - t0:.0f}s")

    # thresholds: bx3 validation (challenge exact match) / Danny val (tile accuracy)
    tile_thr = {}
    Xv, Av, tv, _ = bx3_challenges(specs, "validation")
    Pv = sig(mlp(tile, Xv.reshape(-1, Xv.shape[-1]))).reshape(len(Xv), 9, -1)
    for k, c in enumerate(TILE_CLASSES):
        sel = tv == c
        if sel.sum() >= 30:
            t, em = tune_threshold_challenges(Pv[sel, :, k], Av[sel])
            tile_thr[c] = t
            print(f"  thr {c:14s} {t:.2f}  val exact {em:.4f} (n={sel.sum()})")
    Xd, Yd, Md = tile_dataset(specs, None, "val")
    Pd = sig(mlp(tile, Xd))
    for k, c in enumerate(TILE_CLASSES):
        if c in tile_thr:
            continue
        m = Md[:, k] > 0
        if Yd[m, k].sum() >= 3:
            t, acc = tune_threshold_tiles(Pd[m, k], Yd[m, k] > 0)
            tile_thr[c] = float(np.clip(t, 0.3, 0.7))
            print(f"  thr {c:14s} {tile_thr[c]:.2f}  danny-val tile acc {acc:.4f} (pos={int(Yd[m, k].sum())})")
        else:
            tile_thr[c] = 0.5

    import torch

    def tt(layers):
        return [(torch.from_numpy(W), torch.from_numpy(b)) for W, b in layers]

    if a.tile_only:
        torch.save({"version": 1, "backbones": specs, "classes": TILE_CLASSES, "tile": tt(tile),
                    "tile_thr": tile_thr, "trained_on": "tile head only"}, a.out)
        print(f"saved {a.out} (tile head only)")
        return

    # ---- cell head (4x4)
    cspecs = a.cell_backbones or specs
    E, A4, t4, _ = bx4_challenges(cspecs, "train")
    counts = {c: int((t4 == c).sum()) for c in set(t4) if c}
    cell_classes = sorted(c for c, n in counts.items() if n >= MIN_CELL_CHALLENGES)
    print("4x4 train challenges per class:", counts, "-> cell classes", cell_classes)
    keep = np.isin(t4, cell_classes)
    E, A4, t4 = E[keep], A4[keep], t4[keep]
    K4 = len(cell_classes)
    ci4 = {c: i for i, c in enumerate(cell_classes)}

    def cell_xy(E, A, t):
        Xc = cell_inputs(E).reshape(-1, E.shape[-1] * 3)
        Yc = np.zeros((len(Xc), K4), np.float32)
        Mc = np.zeros((len(Xc), K4), np.float32)
        k = np.repeat([ci4[c] for c in t], 16)
        Yc[np.arange(len(Xc)), k] = A.reshape(-1)
        Mc[np.arange(len(Xc)), k] = 1.0
        return Xc, Yc, Mc

    t1 = time.time()
    cell = train_mlp(*cell_xy(E, A4, t4), hidden=a.cell_hidden, epochs=a.cell_epochs)
    print(f"cell head trained in {time.time() - t1:.0f}s")

    # out-of-fold stage-1 logits for the context stage
    folds = 3
    rng = np.random.default_rng(0)
    fold = rng.integers(0, folds, len(E))
    oof = np.zeros((len(E), 16), np.float32)
    for f in range(folds):
        tr, te = fold != f, fold == f
        part = train_mlp(*cell_xy(E[tr], A4[tr], t4[tr]), hidden=a.cell_hidden, epochs=a.cell_epochs, seed=f + 1)
        L = mlp(part, cell_inputs(E[te]).reshape(-1, E.shape[-1] * 3)).reshape(-1, 16, K4)
        oof[te] = np.take_along_axis(L, np.array([ci4[c] for c in t4[te]])[:, None, None].repeat(16, 1), axis=2)[..., 0]

    def ctx_xy(L, A, t):
        C = context_inputs(L).reshape(-1, 7)
        oh = np.zeros((len(C), K4), np.float32)
        oh[np.arange(len(C)), np.repeat([ci4[c] for c in t], 16)] = 1
        return np.concatenate([C, oh], 1), A.reshape(-1, 1).astype(np.float32), np.ones((len(C), 1), np.float32)

    ctx = train_mlp(*ctx_xy(oof, A4, t4), hidden=32, epochs=20, dropout=0.0, wd=1e-4, lr=3e-3)

    def cell_probs(E, t):
        L = mlp(cell, cell_inputs(E).reshape(-1, E.shape[-1] * 3)).reshape(-1, 16, K4)
        L = np.take_along_axis(L, np.array([ci4[c] for c in t])[:, None, None].repeat(16, 1), axis=2)[..., 0]
        Xc, _, _ = ctx_xy(L, np.zeros_like(L, bool), t)
        return sig(mlp(ctx, Xc)).reshape(-1, 16), sig(L)

    Ev, Av4, tv4, _ = bx4_challenges(cspecs, "validation")
    keep = np.isin(tv4, cell_classes)
    Ev, Av4, tv4 = Ev[keep], Av4[keep], tv4[keep]
    P2, P1 = cell_probs(Ev, tv4)
    cell_thr = {}
    for c in cell_classes:
        sel = tv4 == c
        if sel.sum() >= 20:
            t, em = tune_threshold_challenges(P2[sel], Av4[sel])
            t1_, em1 = tune_threshold_challenges(P1[sel], Av4[sel])
            cell_thr[c] = t
            print(f"  cell thr {c:14s} {t:.2f} val exact {em:.4f} (stage1 only {em1:.4f}, n={sel.sum()})")
        else:
            cell_thr[c] = 0.5

    head = {
        "version": 1,
        "backbones": specs,
        "cell_backbones": cspecs,
        "classes": TILE_CLASSES,
        "tile": tt(tile),
        "tile_thr": tile_thr,
        "cell": tt(cell),
        "cell_ctx": tt(ctx),
        "cell_classes": cell_classes,
        "cell_thr": cell_thr,
        "trained_on": "bxding/recaptcha-67k train (3x3 %d tiles labelled, 4x4 %d challenges) + DannyLuna 57k%s; %s"
                      % (int((M.sum(1) > 0).sum()), len(E), " (off)" if a.no_danny else "",
                         time.strftime("%Y-%m-%d")),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(head, a.out)
    print(f"saved {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
