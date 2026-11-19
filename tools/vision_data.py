"""Shared dataset helpers for the vision tools (grids on disk, labels, embedding cache)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRIDS = ROOT / "data" / "grids"
TILE_LABELS = GRIDS / "tile_labels.json"   # {grid_id: [class name per tile]}
LABELS = GRIDS / "labels.json"             # {grid_id: [selected tile indices]}  (derived)
AUTO_LABELS = GRIDS / "auto_labels.json"   # {grid_id: [class per tile]} model consensus
SPLIT = GRIDS / "split.json"               # record of the val/test grid ids (hash rule)
CACHE = ROOT / "data" / "emb_cache"


def list_grids() -> list[str]:
    return sorted(p.parent.name for p in GRIDS.glob("*/meta.json"))


def meta(gid: str) -> dict:
    return json.loads((GRIDS / gid / "meta.json").read_text())


def target(gid: str) -> str:
    t = meta(gid)["target"]
    return t[0] if isinstance(t, list) else t


def tiles(gid: str) -> list[bytes]:
    return [(GRIDS / gid / f"tile_{i}.jpg").read_bytes() for i in range(9)]


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=0, sort_keys=True))
    tmp.replace(path)


def tile_labels() -> dict[str, list[str]]:
    """Manually reviewed tile classes (the only labels used for val/test scoring)."""
    return load_json(TILE_LABELS, {})


def auto_labels() -> dict[str, list[str]]:
    """Consensus labels from several zero-shot backbones (training only)."""
    return load_json(AUTO_LABELS, {})


def train_labels() -> dict[str, list[str]]:
    """Manual labels take precedence over consensus labels."""
    return {**auto_labels(), **tile_labels()}


def grid_answer(gid: str, classes: list[str]) -> list[int]:
    t = target(gid)
    return [i for i, c in enumerate(classes) if c == t]


def write_derived_labels() -> None:
    tl = tile_labels()
    save_json(LABELS, {g: grid_answer(g, c) for g, c in tl.items()})


def split_of(gid: str) -> str:
    """Deterministic split by hash of the grid id: 30% test, 15% val, 55% train.

    The rule never changes, so a grid can never move between splits as data is added.
    """
    h = int(hashlib.sha1(("split:" + gid).encode()).hexdigest(), 16) % 100
    return "test" if h < 30 else "val" if h < 45 else "train"


def splits(require_manual_eval: bool = True) -> dict[str, list[str]]:
    """{train, val, test} grid ids that have labels.

    train uses manual + consensus labels; val/test only manually reviewed grids (when
    ``require_manual_eval``), so the reported accuracy never depends on model-made labels.
    """
    manual = tile_labels()
    anyl = train_labels()
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for g in sorted(anyl):
        s = split_of(g)
        if s == "train" or not require_manual_eval or g in manual:
            out[s].append(g)
    save_json(SPLIT, {k: v for k, v in out.items() if k != "train"})
    return out


def frozen_split() -> tuple[list[str], list[str]]:
    """Backward compatible (train, test) pair."""
    sp = splits()
    return sp["train"], sp["test"]


def cache_path(spec: str, tta: bool) -> Path:
    return CACHE / (spec.replace(":", "__").replace("/", "_") + ("__tta" if tta else "") + ".npz")


def embeddings(engine, gids: list[str], spec_index: int = 0) -> dict[str, np.ndarray]:
    """Return {grid_id: (9, d)} embeddings for one backbone of ``engine``, cached on disk."""
    spec = engine.specs[spec_index]
    path = cache_path(spec, engine.tta)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = dict(np.load(path)) if path.is_file() else {}
    missing = [g for g in gids if g not in cache]
    if missing:
        from aws_capsolver.vision.grid import decode_tiles
        bb = engine.backbones[spec_index]
        for k in range(0, len(missing), 8):
            chunk = missing[k:k + 8]
            imgs = [im for g in chunk for im in decode_tiles(tiles(g))]
            f = engine._embed_one(bb, imgs)
            for j, g in enumerate(chunk):
                cache[g] = f[9 * j:9 * j + 9]
            print(f"  embedded {min(k + 8, len(missing))}/{len(missing)} with {spec}", file=sys.stderr)
        np.savez(path, **cache)
    return {g: cache[g] for g in gids}
