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
SPLIT = GRIDS / "split.json"               # {"test": [grid ids]} frozen held-out split
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
    return load_json(TILE_LABELS, {})


def grid_answer(gid: str, classes: list[str]) -> list[int]:
    t = target(gid)
    return [i for i, c in enumerate(classes) if c == t]


def write_derived_labels() -> None:
    tl = tile_labels()
    save_json(LABELS, {g: grid_answer(g, c) for g, c in tl.items()})


def frozen_split(test_frac: float = 0.3) -> tuple[list[str], list[str]]:
    """Deterministic split by hash of the grid id, frozen in split.json once written.

    New grids added later are assigned by the same hash rule, so earlier test grids never
    move into training.
    """
    labelled = sorted(tile_labels())
    test = []
    for g in labelled:
        h = int(hashlib.sha1(("split:" + g).encode()).hexdigest(), 16) % 1000
        if h < test_frac * 1000:
            test.append(g)
    frozen = load_json(SPLIT, {"test": []})
    test_set = set(frozen["test"]) | set(test)
    save_json(SPLIT, {"test": sorted(test_set)})
    train = [g for g in labelled if g not in test_set]
    return train, [g for g in labelled if g in test_set]


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
        from aws_capsolver.vision.grid import _decode
        bb = engine.backbones[spec_index]
        for k in range(0, len(missing), 8):
            chunk = missing[k:k + 8]
            imgs = [im for g in chunk for im in _decode(tiles(g))]
            f = engine._embed_one(bb, imgs)
            for j, g in enumerate(chunk):
                cache[g] = f[9 * j:9 * j + 9]
            print(f"  embedded {min(k + 8, len(missing))}/{len(missing)} with {spec}", file=sys.stderr)
        np.savez(path, **cache)
    return {g: cache[g] for g in gids}
