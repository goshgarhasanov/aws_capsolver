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

## General model (`general.py`)

One reader for every common kind of text captcha: digits / letters / mixed, 3-10 (up to 12)
characters, noise lines and curves, dots, grids, strike-through, warped / rotated /
overlapping glyphs, coloured or textured backgrounds, light-on-dark, case-sensitive variants
(CodeIgniter, Securimage, PHP GD, python-`captcha`, BotDetect-like, SimpleCaptcha, ...).

```python
from aws_capsolver.text.general import solve
text, confidence = solve(png_bytes)                                   # ("w8bnx", 0.998)
text, confidence = solve(png_bytes, charset="digits", min_len=5, max_len=6)
text, confidence = solve(png_bytes, charset="0123456789ABCDEF", case_sensitive=True)
```

* `charset`: `None`, `"digits"`, `"letters"`, `"alnum"`, `"lower"`, `"upper"` or an explicit
  string. It masks the logits before decoding (constrained decode, not a post-filter).
* `min_len` / `max_len`: the prefix beam search is pruned at `max_len`; when no beam fits the
  range, an exact length-constrained Viterbi decode supplies the answer.
* `case_sensitive=False` (default) merges the probability of both cases of each letter; the
  returned text keeps the case the model reads (compare it case-insensitively).
* `InvalidImage` for empty / undecodable / tiny / > 2000 px input, `ModelUnavailable` when
  neither engine can load. Lazy thread-safe singleton, deterministic; `warmup()`.
* Weights: `weights/general_ocr.pt` (12.8 MB, `weights_only=True`), override with
  `GENERAL_OCR_WEIGHTS`. ddddocr runs in its own single-threaded ONNX session
  (`GENERAL_OCR_ORT_THREADS`).

### How it works

* **Ours**: CRNN/CTC (`general_model.py`, 2.7 M parameters): RGB 64x256 input, residual CNN
  (stride 8 horizontally) -> 32 frames -> 2-layer BiLSTM -> blank + `0-9a-zA-Z`.
* **Ensemble**: both engines (ours and ddddocr's ONNX model mapped onto the same alphabet)
  propose candidates; every candidate is rescored by both with the exact CTC likelihood and
  the weighted sum (product of experts) wins. Weights and a logistic confidence calibration
  are fitted on validation slices (never test) and stored in the weights file
  (`tools/general_bench.py fit`). On validation ddddocr gets weight 0.05: it only breaks
  near-ties, its contribution is +0.3 points. Calibration on 2088 validation reads:
  confidence >= 0.99 -> 99.9 % correct, 0.95-0.99 -> 97 %, 0.8-0.95 -> 89 %, < 0.5 -> 25 %.
* TrOCR (`anuashok/ocr-captcha-v3`, 334 M parameters) was evaluated but not ensembled:
  4-7 s per image on this CPU (budget 300 ms) and weaker than ours everywhere.

### Training data

`tools/general_synth.py` generates captchas on the fly in 8 style families (general per-char
fonts with a random stack of noise / warps / backgrounds / inversion / blur / JPEG,
python-`captcha` ImageCaptcha with random fonts, claptcha-like, PHP GD bitmap font,
CodeIgniter spiral + sine baseline, Securimage wave warp + arcs, SimpleCaptcha outlines,
BotDetect-like chess / mosaic / fragment / stripes / jail), using ~200 system and OFL fonts.
Real data: the train splits of the sources below. Sources whose labels do not follow the
glyph case (`fold_train`) are trained with a case-folded CTC loss.

Training: 2.6k steps synthetic-heavy pre-training, then 15k steps x 96 on everything
(35 % synthetic, sources weighted by sqrt(size)), AdamW + cosine, EMA weights; CPU only.

### Sources and licences (datasets are never committed; `data/text_general/`)

| source | origin | licence | kind | test n |
|---|---|---|---|---|
| captcha_images_v2 | https://github.com/AakashKumarNain/CaptchaCracker (captcha_images_v2.zip, the Kaggle 'captcha-version-2-images' set by R. Wilhelmy) | unspecified (public Kaggle dataset; research use) | real, 5 lower-case/digit chars, grey, noise line + blur | 300 |
| project_sloth | hf:project-sloth/captcha-images | WTFPL | synthetic, 6 digits, python-captcha-like | 500 |
| recaptcha_v1 | hf:swargaraj/recaptcha-v1-captchas | other (see dataset LICENSE.txt; research use) | real, reCAPTCHA v1 words, 8-10 lower-case letters, warped | 1000 |
| simplecaptcha | hf:muddokon/outlined-text-captcha | MIT | real, SimpleCaptcha (Java) outlined glyphs on gradient, 5 chars | 300 |
| gib_tr | hf:Tuna0000/gib-captcha-labeled | Apache-2.0 | real, Turkish revenue administration (GIB), 6 upper-case/digit chars | 500 |
| ekyb_vn | hf:dathuynh1108/captcha-to-text | unspecified (public HF dataset) | real, Vietnamese government portals (two sites) | 500 |
| hansda | hf:Hansda/captcha-dataset | Apache-2.0 | synthetic, distorted per-char fonts, warps, lines, 4-7 chars | 500 |
| lumasik | hf:lumasik/captcha-25k | MIT | synthetic, 12+ noise filters (blur, salt/pepper, ink bleed), upper/digits | 500 |
| captchaocr500k | hf:AvinashRicky/CaptchaOCR-500K (test split only) | MIT | synthetic, python-captcha-like, labels lower-cased (glyphs are not) | 500 |
| ormeshein | hf:ormeshein/captcha | MIT | synthetic, python-captcha ImageCaptcha look, 4 upper/digit chars | 300 |
| botdetect | hf:squirelmail/dataset-BotDetect-CAPTCHA-Generator (dataset_1000_v2) | Apache-2.0 | synthetic, BotDetect CAPTCHA, 60 styles, 5 upper/digit chars | 1200 |
| nj_biergarten | hf:phunc20/nj_biergarten_captcha (shard 0 of 3) | GPL-3.0 | real, crawled site captcha, 6 chars [a-z1-9] | 1000 |
| yusuf802 | hf:yusuf802/captcha_dataset (shard 0) | unspecified (public HF dataset) | synthetic, coloured / light-on-dark, curves, mixed case, 5 chars | 500 |
| cybertruck | hf:cybertruck32489/captcha_50k (shard 0) | unspecified (public HF dataset) | synthetic, strike lines on textured grey, varied fonts, 5 lower/digit | 500 |
| pycaptcha_default | generated: python `captcha` 0.7.1 ImageCaptcha, default font | BSD (library); images generated locally | synthetic, python-captcha default look, 4-6 mixed-case chars | 1500 |
| local | the user's own samples (CaptchaBypasser/samples, BlsPolandBott*/temp) | private, never committed | real, live site captchas (labelled by eye) | 20 |

Test splits are frozen (official test split where one exists, else a hash of the sample key)
and never trained on. `local` (the user's own BLS Poland and CaptchaBypasser samples, labelled
by eye) and `pycaptcha_default` (python `captcha` default look) have no training data at all.
Excluded: gated datasets (GDSC-NTNU, techietrader, Nischay103, kozhikov), unlabelled ones
(MapleStory), click / behavioural sets.

### Results (2026-11-21, exact match, case-insensitive unless the source is case-sensitive)

| source | kind | n | ddddocr | common | TrOCR* | ours | ensemble | ens.+hints | ens. ms |
|---|---|---|---|---|---|---|---|---|---|
| captcha_images_v2 | real | 300 | 86.3 | 91.0 | 42.2 (n=64) | 98.7 | 98.7 | 99.3 | 91 |
| project_sloth | synthetic | 500 | 65.8 | 69.8 | 43.8 (n=64) | 97.4 | 97.6 | 97.8 | 82 |
| recaptcha_v1 | real | 1000 | 2.3 | 2.1 | 3.1 (n=32) | 40.2 | 39.8 | 42.5 | 77 |
| simplecaptcha | real | 300 | 64.0 | 65.7 | 28.1 (n=64) | 86.0 | 87.0 | 87.0 | 86 |
| gib_tr | real | 500 | 19.6 | 21.4 | 54.7 (n=64) | 94.8 | 95.2 | 94.8 | 69 |
| ekyb_vn | real | 500 | 72.8 | 69.8 | 40.6 (n=32) | 92.2 | 92.8 | 92.8 | 82 |
| hansda | synthetic | 500 | 12.4 | 19.6 | 6.2 (n=64) | 48.0 | 49.0 | 49.0 | 109 |
| lumasik | synthetic | 500 | 26.6 | 35.6 | 28.1 (n=64) | 83.8 | 84.0 | 84.0 | 82 |
| captchaocr500k | synthetic | 500 | 6.6 | 11.8 | 14.1 (n=64) | 67.8 | 68.8 | 68.8 | 84 |
| ormeshein | synthetic | 300 | 33.0 | 36.3 | 21.9 (n=32) | 82.0 | 81.3 | 81.3 | 87 |
| botdetect | synthetic | 1200 | 45.9 | 46.2 | 43.8 (n=32) | 92.7 | 92.8 | 92.8 | 113 |
| nj_biergarten | real | 1000 | 54.8 | 73.2 | 40.6 (n=32) | 99.9 | 99.9 | 99.9 | 80 |
| yusuf802 | synthetic | 500 | 11.4 | 8.8 | 0.0 (n=32) | 60.4 | 60.8 | 60.8 | 88 |
| cybertruck | synthetic | 500 | 85.0 | 88.0 | 34.4 (n=32) | 93.6 | 93.8 | 93.8 | 106 |
| pycaptcha_default | synthetic | 1500 | 5.5 | 4.4 | 0.0 (n=32) | 36.3 | 36.6 | 36.6 | 75 |
| local | real | 20 | 0.0 | 45.0 | 35.0 | 95.0 | 95.0 | 95.0 | 77 |

`ddddocr` = plain `DdddOcr().classification()`; `common` = this package's `common.py`
(ddddocr + preprocessing vote); TrOCR on the first 32-64 test images only (too slow);
`ens.+hints` passes the source's charset / length hints. `pycaptcha_default` is scored
case-sensitively and its default font draws o/O, z/Z, s/S, l/I nearly identically, so much of
its gap is case ambiguity, not misreading. `local` = 11 BLS Poland + 9 CaptchaBypasser samples. Latency: ms per image, mean, ensemble
incl. decoding, torch 2 threads, shared 32-thread laptop CPU.

Honest assessment: the >= 98 % target is reached on captcha_images_v2 (98.7, 99.3 with hints)
and nj_biergarten (99.9) and nearly on project_sloth (97.6). Real site styles with some label
noise or few training images are at 87-95 % (GIB 95.2, local 95.0 = 19/20, cybertruck 93.8,
BotDetect 92.8 over 60 styles, Vietnamese portals 92.8, SimpleCaptcha 87.0 with 800 training
images). Not reached: reCAPTCHA v1 (pseudo-words, heavy warp, 8-10 letters: 40 %), and the
heavily distorted synthetic sets (hansda 49: case-ambiguous per-character sizes; captchaocr500k
69: lower-cased labels over mixed-case glyphs; yusuf802 61, lumasik 84, ormeshein 81: extreme
noise and overlap). Every engine is far lower on these; the small CPU model (2.7 M parameters,
15k steps) is the limit, a larger encoder and longer training are the next step. The ensemble
over ours alone is worth only +0-1 point; ddddocr mainly helps by breaking near-ties.

### Retraining

```bash
.venv/bin/python tools/general_data.py fonts                 # OFL fonts for the generator
.venv/bin/python tools/general_data.py download && .venv/bin/python tools/general_data.py build
.venv/bin/python tools/general_train.py --steps 16000 --synth 0.6 \
    --ckpt data/text_general/ckpt/pre.pt                     # synthetic-heavy warm-up (resumable)
.venv/bin/python tools/general_train.py --init data/text_general/ckpt/pre.pt --steps 15000 \
    --lr 1e-3 --warmup 300 --synth 0.35 --ckpt data/text_general/ckpt/main.pt
.venv/bin/python tools/general_train.py --ckpt data/text_general/ckpt/main.pt \
    --export weights/general_ocr.pt
.venv/bin/python tools/general_bench.py fit                  # ensemble weights + calibration
.venv/bin/python tools/general_bench.py run && .venv/bin/python tools/general_bench.py table
```
