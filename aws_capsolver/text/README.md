# Text captcha recognisers

## BLS Poland (`bls_poland.py`)

Dedicated offline reader for the login captcha of appointment.blspolandvisa.com.

```python
from aws_capsolver.text.bls_poland import solve, solve_many
text, confidence = solve(jpeg_bytes)          # ("56107", 0.997)
```

* Always returns exactly 5 digits (CTC best path constrained to 5 labels) and a confidence
  in 0..1 (product of the per-digit peak posteriors; a blank image scores < 0.5).
* `InvalidImage` (ValueError) for empty / undecodable / oversized input (> 2 MB, > 1024 px
  side), `ModelUnavailable` (RuntimeError) when torch or the weights are missing.
* Lazy thread-safe singleton; deterministic; `warmup()` loads it up front.
* Weights: `weights/bls_poland_ocr.pt` (~4.4 MB, loaded with `weights_only=True`);
  override with `AWS_CAPSOLVER_BLS_POLAND_WEIGHTS`. Inference threads:
  `AWS_CAPSOLVER_TEXT_THREADS` (default 2).

### The captcha

CodeIgniter 3 `create_captcha()` output (GD, `imagejpeg` default quality): 160x55, white
background, the helper's 20-ring line "spiral" in red (centre below the image), 5 black
digits in Open Sans Regular (20 px high), upright, pen advance ~23 px, first x in
[0, 48], each digit's baseline random in [27, 52]. Verified by measuring the real samples
(glyph boxes, spacing fit, colours) and comparing generated images side by side.

### Model and data

* `model.py`: CRNN (6 conv layers + BiGRU 128, 1.1 M params) + CTC over 40 columns.
* `tools/text_synth.py`: faithful port of the CI3 generator (the spiral loop is copied
  1:1), plus nuisance jitter (font Open Sans/Noto/Adwaita, ±per-glyph scale, advance
  21.5-24 px, line colour, JPEG quality 60-90, blur, a second spiral).
  Needs `data/text/fonts/OpenSans-Regular.ttf` (OFL; not versioned).
* `tools/text_data.py`: real samples in `data/text/real/` (gitignored), labels in
  `tools/text_real_labels.csv`; frozen split: TEST iff `sha1(filename)[0] < 192`.
  The collector (≥ 4 s between requests, ≤ 400 images, never submits) stops on 403;
  the site answered 403 on 2026-11-20, so no extra real images were collected.
* `tools/text_train.py`: synthetic pre-training, then fine-tuning with the augmented real
  TRAIN split mixed in (15 %); `--export` writes the weights file.
* `tools/text_eval.py`: exact-match / char accuracy, Wilson 95 % lower bound, latency
  for ddddocr, the BlsPolandBott ddddocr 16-variant vote, and this model.

### Results (2026-11-20)

Real samples: only **11 unique** images exist (the 31 files in the BlsPolandBott
folders are copies), and the site answered 403 to the collector, so the real test set is
tiny; the Wilson bound reflects that. Labels are by eye (all 11 also agree with the model
and, after mapping 'o'->'0', with ddddocr on 10/11).

| reader | real TEST (n=8) | real ALL (n=11) | faithful synthetic | ms / image |
|---|---|---|---|---|
| ddddocr, single call (digits kept) | 0/8 | 0/11 | 157/200 (78.5 %) | ~7 |
| ddddocr, BlsPolandBott 16-variant vote | 0/8 | 0/11 | 192/200 (96.0 %) | ~130 |
| **this model** (`weights/bls_poland_ocr.pt`) | **8/8** (Wilson95 ≥ 0.676) | **11/11** (≥ 0.741) | **3000/3000** (≥ 0.999) | ~10 (2 threads) |
| this model, synthetic-only (before fine-tune) | 8/8 | 11/11 | 5000/5000 | same |

* ddddocr reads this site's '0' as the letter 'o', so after the digit filter every real
  captcha has fewer than 5 digits and the bot's 5-digit vote returns nothing (0 %). Mapping
  'o'->'0' would give 10/11; the dedicated model needs no such patch.
* Robustness (real TEST images, perturbed): JPEG q30 8/8, random shift ±6/±3 px 40/40,
  Gaussian blur 0.7 8/8, upscaled to 200x69 8/8, greyscale 8/8, 20 % darker 8/8.
* Latency (2 inference threads, 32-core box shared with training jobs, load ~40):
  ~10 ms mean in the eval loop, 25 ms median / 51 ms p95 in the worst measurement;
  `solve_many` batches several images in one forward pass.

Reproduce:

    .venv/bin/python tools/text_train.py --phase synth --steps 3000 --out data/text/ckpt/synth.pt
    .venv/bin/python tools/text_train.py --phase finetune --init data/text/ckpt/synth.pt \
        --steps 600 --lr 2e-4 --out data/text/ckpt/ft.pt
    .venv/bin/python tools/text_train.py --export data/text/ckpt/ft.pt
    .venv/bin/python tools/text_eval.py --split test          # or --split all / --synthetic 3000
