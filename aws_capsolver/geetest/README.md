# aws_capsolver.geetest

Solvers for GeeTest v3 / v4 challenges. Only the models live here; the service task type
(`GeeTestClassification`) calls them through `aws_capsolver/geetest_bridge.py`.

Importing the package is cheap: it does not import numpy, OpenCV, torch or open_clip
until a solver is first called. Every image argument accepts raw bytes or base64 (a
`data:` URL prefix is fine). Every function is deterministic and thread-safe.

## Public API

```python
from aws_capsolver.geetest import (slide_offset, slide_details, click_points, nine_grid, gobang, warmup,
                                   GeeTestError, UnsupportedGeeTest, InvalidGeeTestInput)

slide_offset(background, piece=None, full_background=None, y=None) -> int
slide_details(background, piece=None, full_background=None, y=None) -> SlideResult(x, gap_x, y, score, method)
click_points(image, kind, hints=None) -> [(x, y), ...]          # kind: "icon" | "word" | "space"
nine_grid(image, target, count=None) -> [cell, ...]             # 0-based row-major, count default 3
gobang(board) -> ((row, col), (row, col))                       # picture or v4 `ques` matrix
warmup(background=True)
```

* **Slide contract:** with a piece, `slide_offset` returns the x (in background pixels) where
  the **piece image's left edge** must end up. Both widgets draw the piece image from x=0,
  so this is the drag distance. Without a piece it returns the **left edge of the visible
  gap**. Scale it by `displayed_width / background_width`. The piece and the background
  must have the same scale. Raw v3 pictures (312x160, served as shuffled strips) are
  reassembled automatically.
* `y` = the top of the piece image (`ypos` from v3 `get.php` / v4 `load`, or the slice
  element's CSS top scaled to the picture). It rules out v4's decoy gaps by row.
* Errors: `InvalidGeeTestInput` covers bad base64, an undecodable image, more than 4 MB
  or more than 2048 px, a piece with no opaque part, a bad matrix, kind or count.
  `UnsupportedGeeTest` means the variant is not supported, or the solver found no answer.
  Both subclass `GeeTestError`.

## Methods

| type | method | trained weights |
|---|---|---|
| slide v3/v4 | `fullbg` given: piece shape aligned on \|bg − fullbg\|, refined on the piece's own pixels. Otherwise: **oriented shadow edge**, which correlates the log-brightness gradient with the outward normals of the piece outline (the gap is the picture darkened inside that outline). The top shadow peaks are re-ranked by content similarity (gain-invariant NCC of the piece vs the gap, plain + high-pass), because v4 draws decoy gaps, sometimes with the same shape and in the same row. | none |
| gobang (v4 winlinze) | cells are grouped by shift-tolerant colour statistics. Each group is tried as "empty" and the reading with exactly one valid move wins (GeeTest boards have exactly one). Matrix input uses the rule directly. | none |
| nine (v4) | the prompt icon is recognised by its shape among 86 hand-named prompt icons (`weights/geetest_nine_icons.pt`, built by `tools/geetest_nine_icons.py`) → label → CLIP zero-shot cell scoring (`aws_capsolver.vision.grid` engine, open vocabulary contrast, photo synonyms). Unknown icons fall back to CLIP image similarity. Text targets are used directly. | icon table only |
| icon click (v4) | the 2x2 "LED" texture is removed (coloured phase → clean 150x100). A small U-Net heat map gives the icon centres. A crop encoder and a hint encoder (InfoNCE) produce embeddings, and hints are assigned one-to-one to candidates (`weights/geetest_icon.pt`, 6.8 MB). | synthetic only (`tools/geetest_icon_train.py`) |

## Held-out accuracy (real samples from GeeTest's own demo pages, Nov 2026)

`tools/geetest_bench.py`. Duplicates are counted once. Wilson = 95 % lower bound.

| type | n | accuracy | Wilson 95 % LB | latency (CPU) |
|---|---|---|---|---|
| slide v3, bg+piece (±4 px; truth = fullbg diff) | 220 | 100 % | 98.3 % | median 33 ms |
| slide v3, bg+piece+fullbg | 220 | 100 % | 98.3 % | |
| slide v3, bg only (gap left edge ±4) | 220 | 91.4 % | 86.9 % | |
| slide v4, bg+piece (±4 px) | 110 | 97.3 % | 92.3 % | median 53 ms |
| slide v4, bg+piece+`y` | 110 | 100 % | 96.6 % | |
| gobang, matrix | 100 | 100 % | 96.3 % | <1 ms |
| gobang, picture (100 real boards rendered with their own pieces, 2 looks x 4 scales/JPEG) | 800 | 100 % | 99.5 % | median 19 ms |
| gobang, real widget screenshot | 1 | 1/1 | – | |
| nine (hand-verified test split) | 93 | 97.8 % | 92.5 % | see below |
| icon click v4 (hand-verified, all 3 clicks right) | 24 | 70.8 % | 50.8 % | see below |

The truth for slide v4 comes from `tools/geetest_slide_truth.py`: the piece content is
matched on the per-picture median of other samples. Cases where that truth and the solver
disagreed were checked by eye (`tools/geetest_labels/v4_slide_hand.json`). The nine and
icon labels are in `tools/geetest_labels/`.

Latency: slide and gobang take tens of ms. Nine runs two CLIP backbones on 9 cells: about
0.7–0.8 s idle (8 threads), 2.1 s median while the machine was loaded by training. Icon
click takes about 0.1–0.3 s.

## Not supported (raise `UnsupportedGeeTest`)

* **word** click (v3/v4 Chinese characters). v4 word pictures use a different noise
  texture and artistic fonts, and no recogniser was built.
* **space** click.
* **v3 "direction" click**: the official v3 demo's click challenge shows arrows in the hint
  strip ("click the car facing ↓, then ↙, then →"). It is detected and rejected.
  v3 icon click with real hint icons goes through the icon model, but no real v3 sample of
  that kind was available to measure.

## Data and retraining

All real data comes read-only from GeeTest's public demos (`gt4.geetest.com`,
`demos.geetest.com`), with at least 3 s between requests and at most 400 samples per type,
stored in `data/geetest/` (gitignored). Nothing was ever submitted or verified.

```sh
.venv/bin/python tools/geetest_collect.py --types v4_slide,v3_slide,v4_icon,v4_nine,v4_gobang,v3_click,v4_word --max 300
.venv/bin/python tools/geetest_slide_truth.py            # v4 slide truth.json from picture pools
.venv/bin/python tools/geetest_nine_icons.py             # after naming new prompt icons in LABELS
.venv/bin/python tools/geetest_icon_train.py --steps 5000 --threads 8 --workers 5   # ~45 min CPU
.venv/bin/python tools/geetest_bench.py
```

The icon model is trained only on synthetic pictures. These use posterised photos from the
other GeeTest sets as backgrounds, 3–6 icons from GeeTest's own icon set (581 collected
hint icons) as coloured rotated strokes with drop shadows, then the LED-texture + JPEG round
trip. Real v4 icon pictures are only used for the test.
