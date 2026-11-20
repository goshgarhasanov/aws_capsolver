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
4. Decode both runs in one CTranslate2 batch with faster-whisper `medium.en` (int8, beam 5,
   all 5 beam hypotheses kept) and a 20 s encoder window instead of 30 s. Then normalise
   the text (abbreviations, numbers) and remove the repetition loops that a short window
   can cause.
5. Get a second opinion: decode the same runs with `small.en` and a 10 s window
   (`AWS_CAPSOLVER_AUDIO_SECOND_MODEL`, set it to "" to turn it off).
6. Choose the answer:
   - A single word that both models agree on wins (leading "a"/"the"/"one" are ignored).
   - Otherwise the primary model's most confident single word wins. It is scored by mean
     logprob, with penalties for several words, extra BPE tokens and no_speech.
   - Plural-stem rule: "feelings" becomes "feeling" when the stem is one BPE token and
     another beam hypothesis spells it within 0.5 logprob.
   - If the template is not found, fall back to a whole-clip transcription.

The models load once, lazily, behind a lock. Calls into each model are serialised.

- Env vars: `AWS_CAPSOLVER_AUDIO_MODEL` (a size name or a local CTranslate2 directory),
  `AWS_CAPSOLVER_AUDIO_SECOND_MODEL` (default `small.en`), `AWS_CAPSOLVER_AUDIO_WINDOW`
  (seconds, default 20), `AWS_CAPSOLVER_AUDIO_STEM_MARGIN` (default 0.5) and
  `AWS_CAPSOLVER_AUDIO_THREADS` (default 8).
- First download:
  `python -c "from faster_whisper.utils import download_model as d; d('medium.en'); d('small.en')"`
  stores about 2 GB in `~/.cache/huggingface/hub`. Run with `HF_HUB_OFFLINE=1` after
  that.

**Labels.** There is no oracle, so labels come from the agreement of independent models
on each word cut.
- v1: a word counts when at least 2 of 3 voters agree: large-v3 and distil-large-v3 (both
  with a 30 s window) plus small.en with the stock pipeline.
- v2: v1 plus every word that 2 of 4 voters agree on. The fourth voter is large-v3
  re-run, with word timestamps, on the cut words of all 184 clips where the voters
  disagreed. Quiet words that leaked in from the background are dropped. This is add-only
  and was applied to both splits without looking at the answers. It added 16 words on dev
  (16 clips) and 17 on test (17 clips).
- strict: only large-v3, large-v3 with word timestamps and distil-large-v3. small.en is
  left out because it is also the second-opinion model.

1 held-out clip is labelled by hand. The test split (a hash of the clip id) is frozen;
all tuning used dev. An answer is correct when it is in the clip's label set.

| step | dev v1 | dev v2 | test v1 | test v2 | test strict |
|---|---|---|---|---|---|
| medium.en, 20 s window | 92.9 % | 95.8 % | 94.3 % | 94.3 % | 93.9 % |
| + plural-stem rule (margin 0.5) | 93.5 % | 96.4 % | 94.3 % | 94.3 % | 93.9 % |
| + small.en second opinion | 96.4 % | **98.2 %** | 96.0 % | 96.1 % | 95.2 % |

These are offline simulations over cached per-word decodes. Dev n=168. Test n=227 under v1
and n=228 under v2 and strict.

**Final end-to-end held-out run** (the shipped defaults, labels v2): **95.2 % (217/228)**.
Latency per clip: mean 2.96 s, p95 3.06 s, max 4.43 s (16 threads, both models).

The end-to-end run is 2 clips below the 96.1 % simulation. The simulation reused a small.en
decode cached at a different thread count, and int8 CTranslate2 results vary slightly with
the thread count. The plural-stem rule helps on dev only.

Other variants tried:
- Trimming the end of each word cut lost 2–5 points on dev.
- Letting the second model confirm a non-top hypothesis of the primary lost 1–1.5 points.

**98 % is reached on dev but not on held-out data.** The 11 held-out misses that remain
are:
- garbled cuts where both models are wrong (`whismations`, `reflec`, `disting`);
- background speech fused with the word (`source` for "gender");
- plural or derived forms (`feelings`, `notions`, `constant`);
- answers that are probably right but missing from the label set (`regarding`, `settle`).

The gains have flattened. The next lever would be a better separation of the target voice
from the background (a source-separation model or a speaker-conditioned mask), not more
ASR tweaks.

| model, window (single model, labels v1) | dev | latency per clip, mean |
|---|---|---|
| tiny.en, 10 s | 70.7 % | 0.46 s |
| base.en, 10 s | 80.8 % | 0.64 s |
| small.en, 10 s | 85.6 % | 0.6 s |
| small.en, 30 s | 89.8 % | 1.6 s |
| medium.en, 10 s | 89.9 % | 0.9 s |
| medium.en, 20 s | 92.9 % | 2.5 s |
| medium.en, 30 s | 93.5 % | 3.9 s |
| large-v3-turbo, 30 s | – | 6.8 s |

The tiny.en and base.en dev scores were measured against an earlier version of the dev
labels.

```bash
.venv/bin/python tools/collect_audio.py --max 50       # read-only GET /problem, >= 3 s apart
.venv/bin/python tools/bench_audio.py transcribe --model large-v3 --window 30
.venv/bin/python tools/bench_audio.py wordrun             # large-v3 word timestamps, disputed clips
.venv/bin/python tools/bench_audio.py label --models large-v3-wt large-v3@30s \
    distil-large-v3@30s small.en --union data/audio/labels_v1.json
.venv/bin/python tools/bench_audio.py simulate --primary medium.en@20s --second small.en@10s \
    --margins 0 0.5
.venv/bin/python tools/bench_audio.py eval --model medium.en --window 20 --second small.en \
    --split test
```
