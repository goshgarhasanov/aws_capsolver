# reCAPTCHA v2 image classification

`aws_capsolver.recaptcha` solves the image part of reCAPTCHA v2 and returns a CapSolver
`ReCaptchaV2Classification`-style solution.

```python
from aws_capsolver.recaptcha import classify, normalize_question, warm
from aws_capsolver.recaptcha import RecaptchaError, UnsupportedQuestion, InvalidRecaptchaInput

warm(background=True)                                  # call at service start (~18 s load)
classify(img_bytes, "/m/0k4j")                         # {"type": "multi", "objects": [1, 4, 7], "size": 3}
classify(img_bytes, "Select all squares with motorcycles")  # 4x4 -> {"type": "multi", ..., "size": 4}
classify(tile_bytes, "светофоры", size=1)              # {"type": "single", "hasObject": True, "size": 1}
```

* `image`: raw bytes or base64 (a `data:` URL is fine). It can be a 3x3 grid (9 photos, usually
  300 px), a 4x4 grid (1 photo, usually 450 px) or a single 100 px replacement tile.
* `size`: 1, 3 or 4. When it is `None`, the size is inferred: a side under 170 px is a single
  tile; a side of 400 px or more is 4x4, unless strong seams at the thirds show a 3x3; anything
  else is 3x3.
* `question`: a Freebase MID or text in en/tr/az/ru/uk/es/pt/de/fr/it/pl/zh/ja/ko. The
  `questions.py` table maps it to a class key. Unknown English object names such as
  "guitars" are scored zero-shot; anything else raises `UnsupportedQuestion`.
* Errors: `RecaptchaError` is the base class. Its subclasses are `InvalidRecaptchaInput`
  (also a `ValueError`) and `UnsupportedQuestion`. A missing or broken weight file never
  raises: the engine logs a warning and falls back to the next path below.
* `classify_details()` also returns the per-cell probabilities, the threshold and the
  method used (`head`, `dense` or `zeroshot`).

## Pipeline

| Input | Path | Weights |
|---|---|---|
| 3x3 tile, single tile | ViT-B-32 laion2b @224. Blocks 0-9 are frozen and shared with the AWS grid engine; blocks 10-11 are fine-tuned (the "tail"). An MLP head gives multi-label sigmoid outputs, with one threshold per class tuned for challenge exact match. | `weights/recaptcha_tail.pt` (29 MB), `weights/recaptcha_head.pt` (2 MB) |
| 4x4 grid | Dense MIL head. The photo is encoded at 512 px into 16x16 patch tokens (4x4 patches per cell). These are fused with the 17 crop embeddings at 224 px (16 cells + the whole photo). A 1x1 conv and 2 residual 3x3 convs score every patch per class, and each cell takes the log-sum-exp of its 16 patches. Thresholds are per class. | `weights/recaptcha_dense.pt` (7 MB) |
| 4x4 class without a dense head (e.g. car) | The 16 cells are scored with the tile head. | |
| Unknown class or missing weights | Calibrated zero-shot on frozen CLIP: a logistic regression over 5 similarity features (target prompt ensemble vs. street-scene distractors, the other reCAPTCHA classes, and an open vocabulary of ~1000 ImageNet names). | none |

Classes:

* Tile head: bicycle, bridge, bus, car, chimney, crosswalk, fire_hydrant, motorcycle,
  mountain, palm_tree, parking_meter, stairs, taxi, tractor, traffic_light.
* Dense head: bicycle, bus, crosswalk, fire_hydrant, motorcycle, stairs, taxi, tractor,
  traffic_light.
* Zero-shot only: truck, boat, school_bus, and any open-vocabulary target.

## Data

| Dataset | License | Used for |
|---|---|---|
| bxding/recaptcha-67k (HF), 3x3 | MIT (annotations; source images excluded) | train: 12,000 of 31,775 challenges (every k-th, 108k unique tiles); validation: 1,500 (thresholds); test: 3,977 and review: 300 (held out) |
| bxding/recaptcha-67k, 4x4 | same | train: 9,000 of 18,684; validation: 1,500; test: 2,338 and review: 282 (held out) |
| DannyLuna/recaptcha-57k-images-dataset (HF) | MIT | 56k train tiles in 14 folders (positives for the folder class, plus negatives for unrelated classes); val 1,489 (single-tile eval) |
| live 2captcha demo (data/recaptcha/live) | own capture | held out only: 26 3x3 and 127 single tiles, hand-labelled into `labels_manual.json` |

**Label quality caps the measurable accuracy.** recaptcha-67k labels are machine-made:
Qwen asked per tile for 3x3, SAM 3 masks for 4x4. In its `review` split, a reviewer changed
19% of the 3x3 labels and 29% of the 4x4 labels. For 3x3 bus, Qwen marks random noisy street
tiles as bus (34 false-positive cells in 90 challenges); for hydrants it misses small ones.
The `review` split (reviewed labels) is the most trustworthy number below. `test` is scored
against machine labels.

## Results (2026-11-25, `tools/recaptcha_bench.py --live --e2e 100`)

Per-challenge exact match means every cell is right. LB95 is the Wilson 95% lower bound.

| Set | Exact match | LB95 | Per-tile accuracy | Previous interim head |
|---|---|---|---|---|
| 3x3 review (reviewed labels) | 244/300 = 81.3% | 76.5% | 97.7% | 77.3% |
| 3x3 test (machine labels) | 3261/3977 = 82.0% | 80.8% | 97.7% | 76.9% |
| 3x3 live, hand-labelled | 22/26 = 84.6% | 66.5% | | 80.8% |
| 4x4 review (reviewed labels) | 155/282 = 55.0% | 49.1% | 95.6% | 11.7% |
| 4x4 test (machine labels) | 1274/2336 = 54.5% | 52.5% | 95.4% | 13.7% |
| single tile, Danny val | 8367/8507 = 98.35% | 98.06% | | 98.39% |
| single tile, live hand-labelled | 125/127 = 98.4% | 94.4% | | 97.6% |

3x3 per class on review (test in brackets):

| Class | Review | Test |
|---|---|---|
| bicycle | 90.9% | 93.2% |
| bus | 70.0% | 61.6% |
| car | 80.6% | 78.0% |
| crosswalk | 92.5% | 98.3% |
| fire hydrant | 84.3% | 89.0% |

4x4 per class on test:

| Class | Exact |
|---|---|
| bicycle | 65.0% |
| bus | 59.6% |
| crosswalk | 33.6% |
| fire hydrant | 65.6% |
| motorcycle | 55.6% |
| stairs | 29.1% |
| tractor | 62.5% (n=16) |
| traffic light | 53.0% |
| taxi | 10% (n=10) |

Single tile per class (Danny val): mountain 99.9%, stairs 99.3%, chimney 99.4%, motorcycle
99.4%, bus 99.2%, bicycle 99.1%, palm 98.0%, crosswalk 97.9%, bridge 94.5%, traffic light
89.8%, car 80.3%, and 100% for hydrant, parking meter, taxi and tractor. Car and traffic light
are low partly because Danny "Car" tiles often show only tiny distant cars.

Zero-shot fallback (leave-one-class-out on bx3, per-tile accuracy of the held-out class):

| Class | Per-tile accuracy |
|---|---|
| bicycle | 96% |
| bus | 91% |
| car | 86% |
| crosswalk | 75% |
| fire hydrant | 94% |

The old uncalibrated softmax got 70-90% per tile and 0-40% exact.

Warm latency on this laptop CPU (8 torch threads, other jobs running):

| Input | p50 | p90 |
|---|---|---|
| 3x3 | 275 ms | 318 ms |
| single tile | 87 ms | |
| 4x4 | 330-800 ms | 630-870 ms |

`warm()` takes about 18 s.

Things that did not help (measured):

* ViT-B-16 datacomp at 224: same linear-probe accuracy as B-32 on Danny (88.6%) and 4x slower.
* ViT-B-16 at 112 px: 84.3%.
* 3x3 challenge-context features: +0.5%.
* A 4x4 head on the crop embeddings alone: 37% exact, against 55% for the dense fusion head.
* A dense head without the crop fusion: about 3 points lower on validation.

## Known weaknesses

* **The 3x3 bus class** (62-70%) learns Qwen's false positives on noisy street tiles. Better
  labels would fix more than a bigger model would.
* **4x4 exact match is limited by cell borders.** Cells the object barely touches are where
  SAM and a human reviewer disagree. Stairs and crosswalk are the hardest classes.
* **Heavily stippled or posterized tiles** (low-trust sessions) are the main source of 3x3
  errors in live tests. There is no noise augmentation yet, because tokens are cached
  before the tail.
* Some classes have little or no training data: no 3x3 data for bridge, chimney, mountain,
  palm, stairs, tractor or traffic light (Danny tiles only), and none at all for truck, boat
  or school bus (zero-shot only).
* The MIDs `/m/02wbm9` and `/m/0h8nr` could not be verified, so they are not mapped.

## Reproduce

```bash
# data (HF, anonymous; bx images via plain HTTP, rate-limited ~10/s)
#   data/recaptcha/danny/dataset_cls_full_57k/{train,val}/<Class>/*.jpg
#   data/recaptcha/bx/data/{3x3,4x4}/{train,validation,test,review}/...
B32=ViT-B-32:laion2b_s34b_b79k
.venv/bin/python tools/recaptcha_embed.py --backbone $B32@224 --what danny --split train val
.venv/bin/python tools/recaptcha_embed.py --backbone $B32@224 --what bx3 --split train validation test review
.venv/bin/python tools/recaptcha_embed.py --backbone $B32@224 --what bx4 --split train validation test review
.venv/bin/python tools/recaptcha_embed.py --backbone $B32@512 --what bx4dense --split train validation test review
.venv/bin/python tools/recaptcha_finetune.py cache --splits bx3:train bx3:validation bx3:review bx3:test danny:train danny:val
.venv/bin/python tools/recaptcha_finetune.py train --train bx3:train --epochs 2 --out weights/recaptcha_tail.pt
.venv/bin/python tools/recaptcha_finetune.py embed
.venv/bin/python tools/recaptcha_train.py --backbones "$B32@224+tail" --tile-only --out weights/recaptcha_head.pt
.venv/bin/python tools/recaptcha_train_dense.py --fusion --epochs 12 --out weights/recaptcha_dense.pt
.venv/bin/python tools/recaptcha_bench.py --live --e2e 100
.venv/bin/python -m pytest -q tests/test_recaptcha_models.py
```

On this CPU, the embedding and caching steps take several hours in total; each
fine-tuning epoch takes about 35 min.
