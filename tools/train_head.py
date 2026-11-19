#!/usr/bin/env python3
"""Train the tile-class head on frozen CLIP embeddings of the labelled training grids.

The head is a multinomial logistic regression over the concatenated, L2-normalised
embeddings of the engine's backbones. It is saved to weights/aws_capsolver_head.pt and is
picked up automatically by ``aws_capsolver.vision.grid`` when the backbones / TTA match.

  .venv/bin/python tools/train_head.py [--models A:B,C:D] [--tta] [--l2 1e-3] [--all]

``--all`` also trains on the held-out split (use only for the final shipped head, after the
held-out numbers have been recorded).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vision_data as vd  # noqa: E402


def features(engine, gids: list[str]) -> np.ndarray:
    per_bb = [vd.embeddings(engine, gids, k) for k in range(len(engine.specs))]
    return np.concatenate([np.concatenate([pb[g] for pb in per_bb], axis=1) for g in gids])


def fit(x: np.ndarray, y: np.ndarray, n_cls: int, l2: float, iters: int = 300) -> tuple[np.ndarray, np.ndarray]:
    import torch

    X = torch.from_numpy(x).float()
    Y = torch.from_numpy(y).long()
    W = torch.zeros(n_cls, X.shape[1], requires_grad=True)
    b = torch.zeros(n_cls, requires_grad=True)
    scale = 30.0  # embeddings are unit-norm; a fixed temperature speeds up convergence
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(scale * X @ W.T + b, Y) + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return (scale * W).detach().numpy(), b.detach().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models")
    ap.add_argument("--tta", action="store_true", help="embed tile + mirror image")
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--all", action="store_true", help="train on train+val+test (final shipped head)")
    ap.add_argument("--out", default=str(vd.ROOT / "weights" / "aws_capsolver_head.pt"))
    args = ap.parse_args()

    import torch
    from aws_capsolver.vision.grid import GridEngine, KNOWN_TARGETS

    eng = GridEngine(specs=args.models.split(",") if args.models else None, head_path="none",
                     tta=args.tta)
    sp = vd.splits()
    gids = sp["train"] + sp["val"] + sp["test"] if args.all else sp["train"]
    tl = vd.train_labels()
    classes = list(KNOWN_TARGETS)
    if any(c not in classes for g in gids for c in tl[g]):
        classes.append("other")
    y = np.array([classes.index(c if c in classes else "other") for g in gids for c in tl[g]])
    x = features(eng, gids)
    t0 = time.time()
    W, b = fit(x, y, len(classes), args.l2)
    acc = float(((x @ W.T + b).argmax(1) == y).mean())
    print(f"trained on {len(gids)} grids / {len(y)} tiles, {len(classes)} classes, "
          f"train tile acc {acc:.4f}, {time.time() - t0:.1f}s")
    head = {"backbones": eng.specs, "tta": eng.tta, "classes": classes,
            "W": torch.from_numpy(W.astype(np.float32)), "b": torch.from_numpy(b.astype(np.float32)),
            "threshold": 0.5, "thresholds": {}, "n_train_grids": len(gids),
            "trained_on": "train+val+test" if args.all else "train", "created": int(time.time())}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(head, args.out)
    print(f"saved {args.out} ({Path(args.out).stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
