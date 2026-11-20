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

## Audio CAPTCHA (`audio.py`, `noisyneighbour-en-us`)

`transcribe(audio_b64) -> str` returns the `client_solution`: one word, lower-cased, trimmed,
NFC (captcha.js sends `input.value.toLowerCase().trim().normalize("NFC")`).

Each clip (10 s AAC, 16 kHz) is a man reading Kant at about -15 dB, then a fixed recording
"Type one of the two following words spoken by me", then two words from the same voice
about 12 dB above the background. The user only has to type ONE of the two words.

1. Decode with PyAV, which bundles ffmpeg (no binary, no subprocess). Typed errors
   (`AudioDecodeError`, a subclass of `UnsupportedProblem` and `InvalidInput`) cover bad
   base64, oversize input, a non-audio blob, a clip that is too short or too long, and a
   5 s decode budget.
2. Find the instruction by correlating against an embedded energy envelope. On 396 real
   clips the peak was at least 0.86; elsewhere it was at most 0.5.
3. Cut the loud runs after it: a hysteresis threshold of -9/-14 dB relative to the
   instruction voice, padded with silence.
4. Decode both runs in one CTranslate2 batch with faster-whisper `medium.en` (int8, beam 5)
   and a 20 s encoder window instead of 30 s. Then normalise the text (abbreviations,
   numbers) and remove the repetition loops that a short window can cause.
5. Return the most confident single word, scored by mean logprob with penalties for
   several words, extra BPE tokens and no_speech.
   If the template is not found, fall back to a whole-clip transcription.

The model loads once, lazily, behind a lock. Calls into the model are serialised.

- Env vars: `AWS_CAPSOLVER_AUDIO_MODEL` (a size name or a local CTranslate2 directory),
  `AWS_CAPSOLVER_AUDIO_WINDOW` (seconds, default 20) and `AWS_CAPSOLVER_AUDIO_THREADS`
  (default 8).
- First download:
  `python -c "from faster_whisper.utils import download_model as d; d('medium.en')"`
  stores about 1.5 GB in `~/.cache/huggingface/hub`. Run with `HF_HUB_OFFLINE=1` after
  that.

**Accuracy.** There is no oracle, so labels come from the agreement of independent models
on each word cut: large-v3 and distil-large-v3 (both with a 30 s window) plus small.en.
A word counts when at least 2 of the 3 agree. 1 held-out clip was labelled manually.
The production model is not one of the voters.
An answer is correct when it is in the clip's label set. A clip whose voters agreed on
only one word therefore counts a correct second word as wrong, so these numbers are a
lower bound.

| model, window | dev (n=168) | held-out (n=227) | latency per clip, mean (16 threads, idle box) |
|---|---|---|---|
| tiny.en, 10 s | 70.7 % | – | 0.46 s (loaded box) |
| base.en, 10 s | 80.8 % | – | 0.64 s (loaded box) |
| small.en, 10 s | 85.6 % | – | 0.6 s |
| small.en, 30 s | 89.8 % | – | 1.6 s |
| medium.en, 10 s | 89.9 % | – | 0.9 s |
| **medium.en, 20 s (default)** | 91.1 % | **94.3 % (214/227)** | 2.5 s (p95 3.0 s, max 4.3 s) |
| medium.en, 30 s | 93.5 % | – | 3.9 s |
| large-v3-turbo, 30 s | – | – | 6.8 s |

Dev scores for every row except medium.en at 20 s come from cached per-word decodes. The
tiny.en and base.en dev scores were measured against an earlier version of the dev labels.

Most of the 13 held-out misses fall into three groups:
- a plural "s" from the background joined to the word (`locations`, `notions`, `feelings`);
- a garbled word chosen over the clean one;
- a word that the voters did not agree on, so it is missing from the label set.

The 98 % target has not been reached.

```bash
.venv/bin/python tools/collect_audio.py --max 50       # read-only GET /problem, >= 3 s apart
.venv/bin/python tools/bench_audio.py transcribe --model large-v3 --window 30
.venv/bin/python tools/bench_audio.py label --models large-v3@30s distil-large-v3@30s small.en
.venv/bin/python tools/bench_audio.py eval --model medium.en --window 20 --split test
```
