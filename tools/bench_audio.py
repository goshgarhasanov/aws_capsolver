#!/usr/bin/env python3
"""Label and benchmark the audio CAPTCHA solver (aws_capsolver.vision.audio).

Subcommands
  transcribe --model M [--window W] [--split S]
                           decode every foreground word segment of every clip with model M
                           and cache it in data/audio/runs/<M>[@<W>s].json (W=0: stock
                           faster-whisper pipeline; W>0: the solver's batched W-second path)
  label --models A B [--test-models C D E]
                           build data/audio/labels.json: a segment's word is accepted when
                           at least --min-agree runs produce the same single word; the
                           held-out split can use its own voters (exclude the production
                           model there so the test labels are not biased towards it);
                           data/audio/manual.json {clip: [words]} overrides
  eval --model M [--window W] --split test
                           run the production solver over a split and report exact-match
                           accuracy (answer in the clip's label set) and latency
  show <clip-id>           print everything known about one clip

The held-out split is frozen by hashing the clip id (sha1(id) % 100 < HELD_OUT_PCT), so
newly collected clips fall deterministically into one side.

Reproduce the reported numbers:
    for m in large-v3 distil-large-v3; do
        .venv/bin/python tools/bench_audio.py transcribe --model $m --window 30; done
    .venv/bin/python tools/bench_audio.py transcribe --model small.en
    .venv/bin/python tools/bench_audio.py label \
        --models large-v3@30s distil-large-v3@30s small.en
    .venv/bin/python tools/bench_audio.py eval --model medium.en --window 20 --split test
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
    if args.seg:
        tag += "+" + "+".join(args.seg)
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
                         "logprob": round(c.logprob, 3), "no_speech": round(c.no_speech, 3),
                         "alts": [[t, round(v, 3)] for t, v in c.alternatives]})
        done[cid] = {"score": round(score, 3), "segments": rows,
                     "ms": int((time.monotonic() - t0) * 1000)}
        log.info("[%d] %s %s", i, cid, [r["text"] for r in rows])
        if i % 10 == 0:
            out_path.write_text(json.dumps(done, indent=1))
    out_path.write_text(json.dumps(done, indent=1))


def _load_runs(names):
    return {m: json.loads((RUNS / f"{m}.json").read_text()) for m in names}


def cmd_label(args) -> None:
    """Majority vote per segment; the held-out split may use its own (stronger) model set
    that excludes the production model, so the test labels are not biased towards it."""
    dev_runs = _load_runs(args.models)
    test_runs = _load_runs(args.test_models) if args.test_models else dev_runs
    manual_path = DATA / "manual.json"
    manual = json.loads(manual_path.read_text()) if manual_path.exists() else {}
    base = json.loads(Path(args.union).read_text()) if args.union else {}
    labels, review = {}, []
    for cid in clip_ids():
        split = split_of(cid)
        runs = {m: r for m, r in (test_runs if split == "test" else dev_runs).items()
                if cid in r}
        if len(runs) < args.min_agree:
            continue
        nseg = max(len(r[cid]["segments"]) for r in runs.values())
        words = set()
        for k in range(nseg):
            votes = Counter(A.collapse_repeats(A.normalise(r[cid]["segments"][k]["text"]))
                            for r in runs.values() if k < len(r[cid]["segments"]))
            for text, n in votes.items():
                if n >= args.min_agree and text and len(text.split()) == 1:
                    words.add(text)
        if args.union and cid in base:
            words |= set(base[cid]["words"])
        how = "agreement"
        if cid in manual:
            words, how = set(manual[cid]), "manual"
        labels[cid] = {"words": sorted(words), "split": split, "how": how,
                       "models": sorted(runs)}
        if len(words) < 2 and how != "manual":
            review.append(cid)
    LABELS.write_text(json.dumps(labels, indent=1))
    for split in ("dev", "test"):
        part = {c: v for c, v in labels.items() if v["split"] == split}
        print(f"{split}: {len(part)} clips, {sum(len(v['words']) == 2 for v in part.values())} "
              f"with 2 words, {sum(len(v['words']) == 1 for v in part.values())} with 1, "
              f"{sum(not v['words'] for v in part.values())} unlabelled, "
              f"{sum(v['how'] == 'manual' for v in part.values())} manual")
    for cid in review:
        runs = test_runs if labels[cid]["split"] == "test" else dev_runs
        print("REVIEW", labels[cid]["split"], cid, labels[cid]["words"],
              {m: [s["text"] for s in r[cid]["segments"]] for m, r in runs.items() if cid in r})


def cmd_eval(args) -> None:
    labels = json.loads(LABELS.read_text())
    ids = [c for c, v in labels.items() if v["words"] and
           (args.split == "all" or v["split"] == args.split)]
    if args.limit:
        ids = ids[: args.limit]
    if args.stem_margin is not None:
        A.STEM_MARGIN = args.stem_margin
    solver = make_solver(args)
    if args.second:
        solver.second = A.AudioSolver(args.second, cpu_threads=args.threads,
                                      window_s=args.second_window, seg=solver.seg)
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
    print(f"model={args.model} window={args.window:g}s second={args.second or '-'} "
          f"stem_margin={A.STEM_MARGIN:g} beam={args.beam} threads={args.threads} "
          f"split={args.split}: "
          f"{ok}/{n} = {100 * ok / max(n, 1):.1f}%  latency mean {statistics.mean(times):.2f}s "
          f"p95 {sorted(times)[int(0.95 * (n - 1))]:.2f}s max {max(times):.2f}s")
    for w in wrong:
        print("  WRONG", *w)


def disputed_clips(voters: list[str]) -> list[str]:
    """Clips where the voter runs do not all agree on every segment (or miss a segment)."""
    runs = _load_runs(voters)
    out = []
    for cid in clip_ids():
        rows = [r[cid]["segments"] for r in runs.values() if cid in r]
        if len(rows) < len(runs):
            continue
        n = max(len(r) for r in rows)
        for k in range(n):
            texts = {A.collapse_repeats(A.normalise(r[k]["text"])) if k < len(r) else ""
                     for r in rows}
            if len(texts) > 1 or not all(len(t.split()) == 1 for t in texts):
                out.append(cid)
                break
    return out


def cmd_wordrun(args) -> None:
    """Re-decode the cut words of disputed clips with word timestamps (stock pipeline,
    temperature 0) and keep only the words whose span is loud in the original audio, so
    background speech that leaked into a cut is dropped. Cached as <model>-wt.json."""
    RUNS.mkdir(parents=True, exist_ok=True)
    out_path = RUNS / f"{args.model}-wt.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [c for c in disputed_clips(args.voters) if c not in done]
    log.info("%d disputed clips to re-decode", len(todo))
    solver = A.AudioSolver(args.model, cpu_threads=args.threads, window_s=None)
    multilingual = solver.model.model.is_multilingual
    for i, cid in enumerate(todo):
        x = load_clip(cid)
        segs, score, rel = segments_of(x, solver.seg)
        rows = []
        for seg in segs:
            clip, offset = A.cut_segment(x, seg, rel, solver.seg)
            with solver._lock:
                parts, _ = solver.model.transcribe(
                    clip, beam_size=5, word_timestamps=True, temperature=0.0,
                    condition_on_previous_text=False, vad_filter=False,
                    language="en" if multilingual else None)
                words = [w for p in parts for w in (p.words or [])]
            kept = []
            for w in words:
                a = max(0, int((offset + w.start * A.SR) / A.HOP))
                b = min(len(rel), max(a + 1, int((offset + w.end * A.SR) / A.HOP)))
                if b > a and float(rel[a:b].max()) >= args.keep_db:
                    kept.append(w.word)
            rows.append({"start": round(seg.start / A.SR, 2), "dur": round(seg.seconds, 2),
                         "text": A.normalise(" ".join(kept)),
                         "raw": A.normalise(" ".join(w.word for w in words)),
                         "logprob": 0.0, "no_speech": 0.0})
        done[cid] = {"score": round(score, 3), "segments": rows}
        log.info("[%d/%d] %s %s", i + 1, len(todo), cid, [(r["text"], r["raw"]) for r in rows])
        if i % 5 == 0:
            out_path.write_text(json.dumps(done, indent=1))
    out_path.write_text(json.dumps(done, indent=1))


def _cands(row_list, alts=True):
    return [A.Candidate(A.collapse_repeats(A.normalise(r["text"])), r["logprob"],
                        r["no_speech"], None, 1,
                        [(t, v) for t, v in r.get("alts", [])] if alts else [])
            for r in row_list]


def cmd_simulate(args) -> None:
    """Offline policy check on cached runs (same segmentation required): accuracy of
    choose_answer / prefer_stem / second-opinion agreement against labels.json."""
    from faster_whisper.tokenizer import Tokenizer
    from faster_whisper.utils import download_model
    import tokenizers

    tok = tokenizers.Tokenizer.from_file(
        str(Path(download_model(args.tokenizer_model)) / "tokenizer.json"))
    tokenizer = Tokenizer(tok, False)

    def ntok(text):
        return sum(len(tokenizer.encode(" " + w)) for w in text.split()) or 1

    def common(word):
        return len(tokenizer.encode(" " + word)) == 1

    labels = json.loads(LABELS.read_text())
    prim = _load_runs([args.primary])[args.primary]
    sec = _load_runs([args.second])[args.second] if args.second else None
    for margin in args.margins:
        ok = n = 0
        wrong = []
        for cid, v in labels.items():
            if v["split"] != args.split or not v["words"] or cid not in prim:
                continue
            if sec is not None and cid not in sec:
                continue
            cands = _cands(prim[cid]["segments"])
            for c in cands:
                c.n_tokens = ntok(c.text)
            cands = [A.prefer_stem(c, common, margin) for c in cands]
            others = _cands(sec[cid]["segments"], alts=False) if sec is not None else None
            answer = A.choose_answer(cands, others)
            n += 1
            ok += answer in v["words"]
            if answer not in v["words"]:
                wrong.append((cid, answer, v["words"]))
        print(f"{args.primary} second={args.second or '-'} margin={margin:g} {args.split}: "
              f"{ok}/{n} = {100 * ok / max(n, 1):.1f}%")
        if args.verbose:
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
    p.add_argument("--seg", nargs="*", help="SegParams overrides, e.g. edge_end_s=0.03")
    p.add_argument("--window", type=float, default=0.0,
                   help="encoder window in seconds; 0 = stock faster-whisper pipeline")
    p = sub.add_parser("label")
    p.add_argument("--models", nargs="+", required=True, help="runs voting on dev clips")
    p.add_argument("--test-models", nargs="+", help="runs voting on held-out clips")
    p.add_argument("--union", help="earlier labels.json whose words are kept (add-only)")
    p.add_argument("--min-agree", type=int, default=2)
    p = sub.add_parser("eval")
    p.add_argument("--model", default=A.MODEL_NAME)
    p.add_argument("--split", choices=["dev", "test", "all"], default="test")
    p.add_argument("--threads", type=int, default=A.CPU_THREADS)
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--window", type=float, default=A.WINDOW_S)
    p.add_argument("--seg", nargs="*", help="SegParams overrides, e.g. duck_gain_db=-12")
    p.add_argument("--second", default="", help="second-opinion model, e.g. small.en")
    p.add_argument("--second-window", type=float, default=A.SECOND_WINDOW_S)
    p.add_argument("--stem-margin", type=float, default=None)
    p = sub.add_parser("wordrun")
    p.add_argument("--model", default="large-v3")
    p.add_argument("--voters", nargs="+",
                   default=["large-v3@30s", "distil-large-v3@30s", "small.en"])
    p.add_argument("--threads", type=int, default=28)
    p.add_argument("--keep-db", type=float, default=-8.0)
    p = sub.add_parser("simulate")
    p.add_argument("--primary", required=True)
    p.add_argument("--second", default="")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--margins", type=float, nargs="+", default=[0.0])
    p.add_argument("--tokenizer-model", default="medium.en")
    p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("show")
    p.add_argument("clip")
    args = ap.parse_args()
    {"transcribe": cmd_transcribe, "label": cmd_label, "eval": cmd_eval, "wordrun": cmd_wordrun,
     "simulate": cmd_simulate,
     "show": cmd_show}[args.cmd](args)


if __name__ == "__main__":
    main()
