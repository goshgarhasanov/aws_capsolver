# aws_capsolver.vision

Image solvers for AWS WAF CAPTCHA problems.

| problem type            | function                                   | status |
|-------------------------|--------------------------------------------|--------|
| `gridcaptcha-5-*`       | `classify_grid(images, target) -> [int]`   | implemented (CLIP + trained head) |
| `toycarcity`            | `solve_toycar(image) -> (x, y)`            | stub, raises `UnsupportedProblem` (never served by the sampled endpoint) |
| `noisyneighbour-*`      | `transcribe(audio_b64) -> str`             | separate module (`audio.py`), raises `UnsupportedProblem` when absent |

## Grid solver

* Tiles: 9 JPEGs (100x100), row-major. Targets seen so far: bag, bed, bucket, chair,
  clock, curtain, hat. Every `gridcaptcha-5-*` grid has exactly **5** target tiles, so the
  solver returns the 5 most probable tiles (override with `count=` or `threshold=`).
* Model: frozen open_clip backbones (`DEFAULT_MODELS` in `grid.py`) -> concatenated
  L2-normalised embeddings -> multinomial logistic-regression head
  (`weights/aws_capsolver_head.pt`, a few tens of KiB, versioned in git).
  Targets the head does not know fall back to zero-shot CLIP (or raise `UnknownTarget`
  with `AWS_CAPSOLVER_STRICT_TARGETS=1`).
* The engine is created lazily once per process and is thread-safe; call
  `grid.warmup()` at service start to load it in a background thread.
* Errors: `InvalidInput` (bad base64 / image, wrong tile count, bad target or params),
  `UnknownTarget`, `ModelUnavailable` - all subclasses of `VisionError`.

Backbone weights are downloaded from the Hugging Face hub by open_clip on first use
(~1 GB total for the default pair) and cached under `HF_HOME`.

## Accuracy (1000 grids sampled from one live WAF endpoint, Nov 2026)

Held-out **test** split, 319 grids, every tile labelled by hand; head trained on the train
split only (551 grids), selection = top-5:

| model                                   | grid acc | tile acc |
|-----------------------------------------|----------|----------|
| zero-shot ViT-B-32 (laion2b)            | 99.06 %  | 99.79 %  |
| zero-shot ViT-B-16 (datacomp_xl)        | 99.37 %  | 99.86 %  |
| zero-shot ViT-B-16-SigLIP2              | 99.37 %  | 99.86 %  |
| head on ViT-B-32                        | 100 %    | 100 %    |
| **head on ViT-B-32 + ViT-B-16 (default)** | **100 % (319/319, Wilson 95 % lower bound 98.8 %)** | 100 % |

1333 of the 2871 test tiles have no near-duplicate (cosine > 0.95) in the training grids;
all of them are classified correctly. The tile pool repeats a lot, so the shipped head is
retrained on all 1000 grids (`--all`). Latency of the default pair: about 0.6 s per grid on an
idle 8-thread CPU (estimated from per-backbone timings), 1.5 s measured while the machine
was fully loaded by other jobs.

## Retraining

```bash
# 1. collect grids (read-only GET /problem, >= 3 s apart) into data/grids/<id>/
.venv/bin/python tools/collect_samples.py --max 300

# 2. label
#    train split: consensus of three zero-shot backbones (all agree + exactly 5 targets)
.venv/bin/python tools/label_tool.py consensus \
    --models ViT-B-32:laion2b_s34b_b79k,ViT-B-16:datacomp_xl_s13b_b90k,ViT-B-16-SigLIP2:webli
#    everything else (val/test grids, disagreements): manual review sheets
.venv/bin/python tools/label_tool.py sheets --split val test     # -> data/sheets/sheet_N.png
.venv/bin/python tools/label_tool.py accept N [pos=class ...]    # after viewing sheet N

# 3. train on the train split, score on the untouched held-out split
.venv/bin/python tools/train_head.py
.venv/bin/python tools/bench_vision.py --split val    # tuning
.venv/bin/python tools/bench_vision.py --split test   # report once

# 4. final shipped head: all labelled data
.venv/bin/python tools/train_head.py --all
```

Splits are a fixed hash rule on the grid id (`tools/vision_data.py:split_of`): 30 % test,
15 % val, 55 % train. Val/test are scored only against manually reviewed labels.
