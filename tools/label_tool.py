#!/usr/bin/env python3
"""Semi-automatic tile labelling.

  predict   classify every tile of unlabelled grids into a class (zero-shot or head)
  sheets    render review sheets: tiles grouped by predicted class, numbered
  accept    write labels for every tile on a sheet, applying manual corrections

Workflow: predict -> sheets -> look at each sheet -> accept SHEET [pos=class ...].
Labels go to data/grids/tile_labels.json (class per tile); labels.json (selected
indices per grid) is derived from it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vision_data as vd  # noqa: E402

SHEETS = vd.ROOT / "data" / "sheets"
PSEUDO = vd.GRIDS / "pseudo.json"


def classes() -> list[str]:
    from aws_capsolver.vision.grid import KNOWN_TARGETS
    return list(KNOWN_TARGETS) + ["other"]


def cmd_predict(args) -> None:
    from aws_capsolver.vision.grid import get_engine, KNOWN_TARGETS, DISTRACTORS
    eng = get_engine()
    gids = [g for g in vd.list_grids() if args.all or g not in vd.tile_labels()]
    out = vd.load_json(PSEUDO, {})
    all_cls = list(KNOWN_TARGETS) + list(DISTRACTORS)
    per_bb = [vd.embeddings(eng, gids, k) for k in range(len(eng.specs))]
    for g in gids:
        feats = [pb[g] for pb in per_bb]
        if eng.head is not None:
            x = np.concatenate(feats, 1)
            lg = x @ eng.head["W"].T + eng.head["b"]
            lg -= lg.max(1, keepdims=True)
            p = np.exp(lg); p /= p.sum(1, keepdims=True)
            names = eng.head["classes"]
        else:
            ps = []
            for bb, f in zip(eng.backbones, feats):
                T = np.stack([eng._text(bb, c) for c in all_cls])
                lg = 100 * f @ T.T
                lg -= lg.max(1, keepdims=True)
                q = np.exp(lg); q /= q.sum(1, keepdims=True)
                ps.append(q)
            p = np.mean(ps, 0)
            names = all_cls
        res = []
        for i in range(9):
            j = int(p[i].argmax())
            c = names[j] if names[j] in KNOWN_TARGETS else "other"
            res.append([c, float(p[i, j])])
        out[g] = res
    vd.save_json(PSEUDO, out)
    print(f"predicted {len(gids)} grids")


def _zero_shot_classes(eng, feats: list[np.ndarray]) -> list[list[str]]:
    """Per-backbone argmax over the known classes for 9 tiles.

    Only the known classes are used: every sampled tile belongs to one of them, and tiles
    that do not are caught by backbone disagreement or the 5-targets check.
    """
    from aws_capsolver.vision.grid import KNOWN_TARGETS
    names = list(KNOWN_TARGETS)
    out = []
    for bb, f in zip(eng.backbones, feats):
        T = np.stack([eng._text(bb, c) for c in names])
        idx = (f @ T.T).argmax(1)
        out.append([names[j] if names[j] in KNOWN_TARGETS else "other" for j in idx])
    return out


def cmd_consensus(args) -> None:
    """Label train grids where every backbone agrees on every tile and the target count is 5.

    Grids that fail the check get their majority vote written to pseudo.json for review.
    Val/test grids are never auto-labelled; they always go to manual review.
    """
    from collections import Counter
    from aws_capsolver.vision.grid import GridEngine
    eng = GridEngine(specs=args.models.split(","), head_path="none", tta=False)
    manual = vd.tile_labels()
    gids = [g for g in vd.list_grids() if g not in manual]
    per_bb = [vd.embeddings(eng, gids, k) for k in range(len(eng.specs))]
    auto = vd.auto_labels()
    pseudo = vd.load_json(PSEUDO, {})
    stats = Counter()
    for g in gids:
        votes = _zero_shot_classes(eng, [pb[g] for pb in per_bb])
        maj = [Counter(v[i] for v in votes).most_common(1)[0] for i in range(9)]
        agree = all(n == len(votes) for _, n in maj)
        labels = [c for c, _ in maj]
        ok = agree and "other" not in labels and labels.count(vd.target(g)) == 5
        if ok and vd.split_of(g) == "train":
            auto[g] = labels
            pseudo.pop(g, None)
            stats["auto"] += 1
        else:
            auto.pop(g, None)
            pseudo[g] = [[c, n / len(votes)] for c, n in maj]
            stats["review_" + vd.split_of(g) + ("" if ok else "_disagree")] += 1
    vd.save_json(vd.AUTO_LABELS, auto)
    vd.save_json(PSEUDO, pseudo)
    print(dict(stats))


def cmd_sheets(args) -> None:
    from PIL import Image, ImageDraw
    pseudo = vd.load_json(PSEUDO, {})
    labelled = vd.tile_labels()
    items = []
    for g, res in pseudo.items():
        if g in labelled and not args.relabel:
            continue
        if args.gids and g not in args.gids:
            continue
        if args.split and vd.split_of(g) not in args.split:
            continue
        if args.limit and len({it[2] for it in items} | {g}) > args.limit:
            break
        for i, (c, conf) in enumerate(res):
            items.append((c, conf, g, i))
    items.sort(key=lambda t: (t[0], t[1]))
    SHEETS.mkdir(parents=True, exist_ok=True)
    for old in SHEETS.glob("sheet_*"):
        old.unlink()
    cols, rows, sz = args.cols, args.rows, args.size
    per = cols * rows
    k = 0
    by_cls: dict[str, list] = {}
    for it in items:
        by_cls.setdefault(it[0], []).append(it)
    for c, lst in by_cls.items():
        for s in range(0, len(lst), per):
            chunk = lst[s:s + per]
            W = Image.new("RGB", (cols * (sz + 2), rows * (sz + 2) + 18), "white")
            d = ImageDraw.Draw(W)
            d.text((4, 2), f"sheet {k}: predicted '{c}' ({len(chunk)} tiles)", fill="red")
            mapping = {}
            for n, (_, conf, g, i) in enumerate(chunk):
                x, y = (n % cols) * (sz + 2), 18 + (n // cols) * (sz + 2)
                W.paste(Image.open(vd.GRIDS / g / f"tile_{i}.jpg").resize((sz, sz)), (x, y))
                d.rectangle([x, y, x + 16, y + 10], fill="black")
                d.text((x + 1, y), str(n), fill="yellow")
                mapping[n] = [g, i]
            W.save(SHEETS / f"sheet_{k}.png")
            (SHEETS / f"sheet_{k}.json").write_text(json.dumps({"class": c, "tiles": mapping}))
            k += 1
    print(f"{k} sheets, {len(items)} tiles -> {SHEETS}")


def cmd_accept(args) -> None:
    sheet = json.loads((SHEETS / f"sheet_{args.sheet}.json").read_text())
    fixes = {}
    for f in args.fixes:
        pos, _, c = f.partition("=")
        if c not in classes():
            raise SystemExit(f"unknown class {c!r}; valid: {classes()}")
        fixes[str(int(pos))] = c
    pseudo = vd.load_json(PSEUDO, {})
    tl = vd.tile_labels()
    pending = vd.load_json(vd.GRIDS / "pending_tiles.json", {})
    for pos, (g, i) in sheet["tiles"].items():
        c = fixes.get(pos, sheet["class"])
        pending.setdefault(g, [None] * 9)[i] = c
    # a grid is labelled once all 9 tiles have been reviewed
    done = 0
    for g, lst in list(pending.items()):
        if all(x is not None for x in lst):
            tl[g] = lst
            del pending[g]
            done += 1
    vd.save_json(vd.TILE_LABELS, tl)
    vd.save_json(vd.GRIDS / "pending_tiles.json", pending)
    vd.write_derived_labels()
    print(f"sheet {args.sheet}: {len(fixes)} fixes; {done} grids completed; total labelled {len(tl)}; "
          f"pending {len(pending)}; pseudo {len(pseudo)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict"); p.add_argument("--all", action="store_true")
    s = sub.add_parser("sheets")
    s.add_argument("--cols", type=int, default=12); s.add_argument("--rows", type=int, default=8)
    s.add_argument("--size", type=int, default=72); s.add_argument("--relabel", action="store_true")
    s.add_argument("--gids", nargs="*")
    s.add_argument("--split", nargs="*", choices=["train", "val", "test"])
    s.add_argument("--limit", type=int, help="max number of grids on this batch of sheets")
    c = sub.add_parser("consensus"); c.add_argument("--models", required=True)
    a = sub.add_parser("accept"); a.add_argument("sheet", type=int); a.add_argument("fixes", nargs="*")
    args = ap.parse_args()
    {"predict": cmd_predict, "sheets": cmd_sheets, "accept": cmd_accept,
     "consensus": cmd_consensus}[args.cmd](args)


if __name__ == "__main__":
    main()
