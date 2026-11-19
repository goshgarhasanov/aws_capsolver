#!/usr/bin/env python3
"""Label and benchmark the audio CAPTCHA solver (aws_capsolver.vision.audio).

Subcommands
  transcribe --model M     run model M on every foreground word segment of every clip and
                           cache the result in data/audio/runs/<M>.json
  label --models A B C     build data/audio/labels.json: a segment's word is accepted when
                           at least --min-agree of the given (cached) models produce the
                           same single word; clips with < 2 accepted words go to review
  eval --model M           run the production solver with model M over the frozen held-out
                           split and report exact-match accuracy (answer in the clip's
                           label set) and latency
  show <clip-id>           print everything known about one clip

The held-out split is frozen by hashing the clip id (sha1(id) % 100 < HELD_OUT_PCT), so
newly collected clips fall deterministically into one side.

Usage:
    .venv/bin/python tools/bench_audio.py transcribe --model large-v3
    .venv/bin/python tools/bench_audio.py label --models large-v3 medium.en distil-large-v3
    .venv/bin/python tools/bench_audio.py eval --model small.en --split test
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aws_capsolver.vision import audio as A  # noqa: E402

DATA = ROOT / "data" / "audio"
CLIPS = DATA / "clips"
RUNS = DATA / "runs"
LABELS = DATA / "labels.json"
HELD_OUT_PCT = 60
log = logging.getLogger("bench_audio")


def clip_ids() -> list[str]:
    return sorted(p.parent.name for p in CLIPS.glob("*/audio.aac"))


def split_of(cid: str) -> str:
    return "test" if int(hashlib.sha1(cid.encode()).hexdigest(), 16) % 100 < HELD_OUT_PCT \
        else "dev"


def load_clip(cid: str):
    return A.decode_audio((CLIPS / cid / "audio.aac").read_bytes())


def segments_of(x, params=A.DEFAULT_SEG):
    """(segments, template score, relative dB envelope) for one clip."""
    end, score = A.locate_instruction(x)
    if score < A.MIN_TEMPLATE_SCORE:
        return [], score, None
    ref = A.reference_level(x, end)
    return A.find_word_segments(x, end, ref, params), score, A.relative_db(x, ref)


def make_solver(args) -> A.AudioSolver:
    overrides = {}
    for item in getattr(args, "seg", None) or []:
        key, _, value = item.partition("=")
        overrides[key] = float(value)
    return A.AudioSolver(args.model, cpu_threads=args.threads, beam_size=args.beam,
                         window_s=args.window if args.window > 0 else None,
                         seg=dataclasses.replace(A.DEFAULT_SEG, **overrides))


def cmd_transcribe(args) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    tag = args.model if args.window <= 0 else f"{args.model}@{args.window:g}s"
    out_path = RUNS / f"{tag}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    solver = make_solver(args)
    for i, cid in enumerate(clip_ids()):
        if cid in done or (args.split != "all" and split_of(cid) != args.split):
            continue
        x = load_clip(cid)
        segs, score, rel = segments_of(x, solver.seg)
        t0 = time.monotonic()
        rows = []
        clips = [A.cut_segment(x, s, rel, solver.seg)[0] for s in segs]
        decoded = solver._decode_batch(clips) if solver.window_s else \
            [solver._decode(c) for c in clips]
        for s, c in zip(segs, decoded):
            rows.append({"start": round(s.start / A.SR, 2), "dur": round(s.seconds, 2),
                         "peak": round(s.peak_db, 1), "text": c.text,
                         "logprob": round(c.logprob, 3), "no_speech": round(c.no_speech, 3)})
        done[cid] = {"score": round(score, 3), "segments": rows,
                     "ms": int((time.monotonic() - t0) * 1000)}
        log.info("[%d] %s %s", i, cid, [r["text"] for r in rows])
        if i % 10 == 0:
            out_path.write_text(json.dumps(done, indent=1))
    out_path.write_text(json.dumps(done, indent=1))


def cmd_label(args) -> None:
    runs = {m: json.loads((RUNS / f"{m}.json").read_text()) for m in args.models}
    manual = json.loads((DATA / "manual.json").read_text()) if (DATA / "manual.json").exists() \
        else {}
    labels, review = {}, []
    for cid in clip_ids():
        if not all(cid in r for r in runs.values()):
            continue
        nseg = {len(r[cid]["segments"]) for r in runs.values()}
        words = set()
        for k in range(max(nseg)):
            votes = Counter(A.normalise(r[cid]["segments"][k]["text"]) for r in runs.values()
                            if k < len(r[cid]["segments"]))
            text, n = votes.most_common(1)[0]
            if n >= args.min_agree and len(text.split()) == 1:
                words.add(text)
        if cid in manual:
            words = set(manual[cid])
        labels[cid] = {"words": sorted(words), "split": split_of(cid),
                       "how": "manual" if cid in manual else "agreement"}
        if len(words) < 2:
            review.append(cid)
    LABELS.write_text(json.dumps(labels, indent=1))
    n = len(labels)
    print(f"labelled {n} clips; {n - len(review)} with 2 agreed words; "
          f"{len(review)} need review; usable (>=1 word): "
          f"{sum(1 for v in labels.values() if v['words'])}")
    for cid in review:
        print("REVIEW", cid, labels[cid]["words"],
              {m: [s["text"] for s in r[cid]["segments"]] for m, r in runs.items()})


def cmd_eval(args) -> None:
    labels = json.loads(LABELS.read_text())
    ids = [c for c, v in labels.items() if v["words"] and
           (args.split == "all" or v["split"] == args.split)]
    if args.limit:
        ids = ids[: args.limit]
    solver = make_solver(args)
    solver.solve(load_clip(ids[0]))  # warm-up
    ok, times, wrong = 0, [], []
    for cid in ids:
        x = load_clip(cid)
        t0 = time.monotonic()
        answer = solver.solve(x)
        times.append(time.monotonic() - t0)
        if answer in labels[cid]["words"]:
            ok += 1
        else:
            wrong.append((cid, answer, labels[cid]["words"]))
    n = len(ids)
    print(f"model={args.model} window={args.window:g}s beam={args.beam} threads={args.threads} "
          f"split={args.split}: "
          f"{ok}/{n} = {100 * ok / max(n, 1):.1f}%  latency mean {statistics.mean(times):.2f}s "
          f"p95 {sorted(times)[int(0.95 * (n - 1))]:.2f}s max {max(times):.2f}s")
    for w in wrong:
        print("  WRONG", *w)


def cmd_show(args) -> None:
    for path in sorted(RUNS.glob("*.json")):
        run = json.loads(path.read_text())
        if args.clip in run:
            print(path.stem, json.dumps(run[args.clip]["segments"]))
    if LABELS.exists():
        print("label", json.loads(LABELS.read_text()).get(args.clip))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("transcribe")
    p.add_argument("--model", required=True)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--window", type=float, default=0.0,
                   help="encoder window in seconds; 0 = stock faster-whisper pipeline")
    p = sub.add_parser("label")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--min-agree", type=int, default=2)
    p = sub.add_parser("eval")
    p.add_argument("--model", default=A.MODEL_NAME)
    p.add_argument("--split", choices=["dev", "test", "all"], default="test")
    p.add_argument("--threads", type=int, default=A.CPU_THREADS)
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--window", type=float, default=A.WINDOW_S)
    p.add_argument("--seg", nargs="*", help="SegParams overrides, e.g. duck_gain_db=-12")
    p = sub.add_parser("show")
    p.add_argument("clip")
    args = ap.parse_args()
    {"transcribe": cmd_transcribe, "label": cmd_label, "eval": cmd_eval,
     "show": cmd_show}[args.cmd](args)


if __name__ == "__main__":
    main()
