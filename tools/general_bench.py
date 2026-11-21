"""
Per-source benchmark of the general text-captcha reader and its baselines, plus fitting of
the ensemble weights / confidence calibration.

    # table on the frozen test splits (tools/general_data.py build ...)
    .venv/bin/python tools/general_bench.py run --out data/text_general/bench.json
    .venv/bin/python tools/general_bench.py trocr --trocr 64     # TrOCR on 64 per source
    .venv/bin/python tools/general_bench.py table                # merged markdown table
    # fit the ensemble on the validation slices (never the test splits) and store it
    .venv/bin/python tools/general_bench.py fit --weights weights/general_ocr.pt

Engines in the table:
  ddddocr    plain `DdddOcr().classification()` (what most people run)
  common     aws_capsolver.text.common (ddddocr + preprocessing vote), when present
  trocr      anuashok/ocr-captcha-v3 (TrOCR, Apache-2.0), optional, slow on CPU
  ours       our CRNN alone (constrained beam decode)
  ensemble   general.solve() without hints
  +hints     general.solve() with the source's charset / length hints

Accuracy is exact match, case-insensitive unless the source is case-sensitive.
"""

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from general_data import PACKS, SOURCES, read_pack  # noqa: E402


def _same(a: str, b: str, case: bool) -> bool:
    return a == b if case else a.lower() == b.lower()


class Trocr:
    NAME = "anuashok/ocr-captcha-v3"

    def __init__(self, threads=6):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        torch.set_num_threads(threads)
        self.torch = torch
        self.proc = TrOCRProcessor.from_pretrained(self.NAME)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.NAME).eval()

    def read_batch(self, images):
        from PIL import Image
        ims = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        with self.torch.inference_mode():
            pv = self.proc(ims, return_tensors="pt").pixel_values
            out = self.model.generate(pv, max_new_tokens=16)
        return ["".join(t.split()) for t in self.proc.batch_decode(out, skip_special_tokens=True)]


def run(args):
    import torch
    torch.set_num_threads(args.threads)   # like the service (AWS_CAPSOLVER_TEXT_THREADS)
    from aws_capsolver.text import general
    from aws_capsolver.text import general_model as gm
    engine = general._get_engine()
    import ddddocr
    dddd = ddddocr.DdddOcr(show_ad=False)
    try:
        from aws_capsolver.text import common
        common.warmup()
    except Exception:
        common = None

    names = args.sources.split(",") if args.sources else [
        n for n in SOURCES if (PACKS / n / "test.json").exists()]
    results = {}
    for name in names:
        src = SOURCES[name]
        _, labels, images = read_pack(name, "test")
        if args.limit:
            labels, images = labels[:args.limit], images[:args.limit]
        case = src.case_sensitive
        hint_cs = src.charset
        lo, hi = src.lengths
        stats = {k: [0, 0.0] for k in ("ddddocr", "common", "ours", "ensemble", "hints")}
        errors = []
        for label, data in zip(labels, images):
            t = time.perf_counter()
            r = dddd.classification(data)
            stats["ddddocr"][1] += time.perf_counter() - t
            stats["ddddocr"][0] += _same(r or "", label, case)
            if common is not None:
                t = time.perf_counter()
                try:
                    r = common.recognize(data).text
                except Exception:
                    r = ""
                stats["common"][1] += time.perf_counter() - t
                stats["common"][0] += _same(r, label, case)
            # ours alone
            t = time.perf_counter()
            rgb = general.load_image(data)
            lp = engine.ours.logp(rgb)
            mask = gm.charset_mask(None)
            view, vmask = general._view(lp, mask, case)
            cands = general._candidates(view, vmask, 1, gm.MAX_LABEL_LEN)
            ours = gm.classes_to_text(cands[0]) if cands else ""
            stats["ours"][1] += time.perf_counter() - t
            stats["ours"][0] += _same(ours, label, case)
            t = time.perf_counter()
            text, conf = general.solve(data, case_sensitive=case)
            stats["ensemble"][1] += time.perf_counter() - t
            ok = _same(text, label, case)
            stats["ensemble"][0] += ok
            if not ok and len(errors) < 15:
                errors.append((label, text, round(conf, 3)))
            t = time.perf_counter()
            text, _ = general.solve(data, charset=hint_cs, min_len=lo or None,
                                    max_len=hi or None, case_sensitive=case)
            stats["hints"][1] += time.perf_counter() - t
            stats["hints"][0] += _same(text, label, case)
        n = len(labels)
        row = {"n": n, "kind": src.kind, "case_sensitive": case}
        for k, (ok, dt) in stats.items():
            if k == "common" and common is None:
                continue
            row[k] = round(ok / n, 4)
            row[k + "_ms"] = round(1000 * dt / n, 1)
        row["errors"] = errors
        results[name] = row
        print(name, json.dumps({k: v for k, v in row.items() if k != "errors"}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(table(results))


def run_trocr(args):
    """TrOCR baseline only (slow); stored next to the main results and merged by `table`."""
    trocr = Trocr()
    names = args.sources.split(",") if args.sources else [
        n for n in SOURCES if (PACKS / n / "test.json").exists()]
    path = Path(_trocr_path(args.out))
    out = json.loads(path.read_text()) if path.exists() else {}
    for name in names:
        case = SOURCES[name].case_sensitive
        _, labels, images = read_pack(name, "test")
        m = min(len(labels), args.trocr or 50)
        ok, dt = 0, 0.0
        for i in range(0, m, 16):
            t = time.perf_counter()
            outs = trocr.read_batch(images[i:min(m, i + 16)])
            dt += time.perf_counter() - t
            ok += sum(_same(o, l, case) for o, l in zip(outs, labels[i:i + 16]))
        out[name] = {"trocr": round(ok / m, 4), "trocr_n": m, "trocr_ms": round(1000 * dt / m, 1)}
        print(name, out[name], flush=True)
        path.write_text(json.dumps(out, indent=1))   # merged, written after every source


def _trocr_path(out: str) -> str:
    return str(Path(out).with_name(Path(out).stem + "_trocr.json"))


def table(results) -> str:
    cols = [("ddddocr", "ddddocr"), ("common", "common"), ("trocr", "TrOCR*"),
            ("ours", "ours"), ("ensemble", "ensemble"), ("hints", "ens.+hints")]
    cols = [c for c in cols if any(c[0] in r for r in results.values())]
    head = "| source | kind | n | " + " | ".join(c[1] for c in cols) + " | ens. ms |"
    lines = [head, "|" + "---|" * (len(cols) + 4)]
    for name, r in results.items():
        cells = []
        for k, _ in cols:
            if k not in r:
                cells.append("–")
            elif k == "trocr" and r.get("trocr_n", r["n"]) < r["n"]:
                cells.append(f"{100 * r[k]:.1f} (n={r['trocr_n']})")
            else:
                cells.append(f"{100 * r[k]:.1f}")
        lines.append(f"| {name} | {r['kind']} | {r['n']} | " + " | ".join(cells)
                     + f" | {r.get('ensemble_ms', 0):.0f} |")
    return "\n".join(lines)


# fitting -----------------------------------------------------------------------
def fit(args):
    """Grid-search the engine weights and fit the logistic confidence on val slices."""
    import torch
    from aws_capsolver.text import general
    from aws_capsolver.text import general_model as gm
    from general_train import Pack
    engine = general._get_engine()
    rows = []   # per sample: list of (cand, ll_ours, ll_dddd), label, case, tops
    names = [n for n, s in SOURCES.items() if not s.eval_only and (PACKS / n / "train.json").exists()]
    for name in names:
        src = SOURCES[name]
        p = Pack(name, False, args.val_per_source)
        idx = list(range(p.n_train, len(p.labels)))[:args.per_source]
        for i in idx:
            data = bytes(p.blob[p.offsets[i]:p.offsets[i + 1]])
            label = p.labels[i]
            case = src.case_sensitive
            rgb = general.load_image(data)
            mask = gm.charset_mask(None)
            views = {"ours": general._view(engine.ours.logp(rgb), mask, case)}
            if engine.dddd is not None:
                views["ddddocr"] = general._view(engine.dddd.logp(general._dddd_input(rgb)), mask, case)
            cands, tops = {}, {}
            for k, (v, vm) in views.items():
                c = general._candidates(v, vm, 1, gm.MAX_LABEL_LEN)
                tops[k] = c[0] if c else None
                for x in c:
                    cands.setdefault(x, None)
            sc = []
            for x in cands:
                lls = [general.ctc_log_likelihood(views[k][0], x) if k in views else general.FLOOR
                       for k in ("ours", "ddddocr")]
                lls = [max(general.FLOOR, v) if np.isfinite(v) else general.FLOOR for v in lls]
                sc.append((gm.classes_to_text(x), lls[0], lls[1], x))
            rows.append((sc, label, case, tops))
        print(f"{name}: {len(idx)} val samples", flush=True)

    def pick(sc, w):
        return max(sc, key=lambda s: s[1] + w * s[2]) if sc else None

    def search(subset):
        best_w, best_acc = 0.0, -1.0
        for w in np.round(np.arange(0.0, 1.21, 0.05), 2):
            acc = np.mean([bool(sc) and _same(pick(sc, w)[0], lab, case)
                           for sc, lab, case, _ in subset])
            if acc > best_acc + 1e-9:
                best_w, best_acc = float(w), float(acc)
        return best_w, best_acc

    ci = [r for r in rows if not r[2]]
    cs = [r for r in rows if r[2]]
    best_w, ci_acc = search(ci)
    best_w_cs, cs_acc = search(cs) if cs else (best_w, float("nan"))
    best_acc = (ci_acc * len(ci) + (cs_acc * len(cs) if cs else 0)) / len(rows)
    ours_acc = np.mean([bool(sc) and _same(pick(sc, 0.0)[0], lab, case) for sc, lab, case, _ in rows])
    print(f"w_dddd={best_w} (case-insensitive acc {ci_acc:.4f}), w_dddd_cs={best_w_cs} "
          f"(case-sensitive acc {cs_acc:.4f}); overall {best_acc:.4f}, ours alone {ours_acc:.4f}")
    # logistic calibration
    X, y = [], []
    for sc, lab, case, tops in rows:
        if not sc:
            continue
        wd = best_w_cs if case else best_w
        ranked = sorted(sc, key=lambda s: -(s[1] + wd * s[2]))
        top = ranked[0]
        margin = (top[1] + wd * top[2]) - (ranked[1][1] + wd * ranked[1][2]) \
            if len(ranked) > 1 else 10.0
        agree = 1.0 if tops.get("ours") == tops.get("ddddocr") == top[3] else 0.0
        X.append([top[1], top[2], agree, min(margin, 10.0)])
        y.append(1.0 if _same(top[0], lab, case) else 0.0)
    X = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=500)

    def closure():
        opt.zero_grad()
        z = w[0] + X @ w[1:]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + 1e-4 * (w[1:] ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    calib = [round(float(v), 5) for v in w.detach()]
    with torch.no_grad():
        prob = torch.sigmoid(w[0] + X @ w[1:])
        # reliability: accuracy in confidence bins
        for lo_, hi_ in ((0, .5), (.5, .8), (.8, .95), (.95, .99), (.99, 1.01)):
            sel = (prob >= lo_) & (prob < hi_)
            if sel.any():
                print(f"conf {lo_:.2f}-{min(hi_, 1):.2f}: n={int(sel.sum())} "
                      f"acc={float(y[sel].mean()):.3f} mean_conf={float(prob[sel].mean()):.3f}")
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    state["ensemble"] = {"w_ours": 1.0, "w_dddd": best_w, "w_dddd_cs": best_w_cs, "calib": calib,
                         "val_acc": round(float(best_acc), 4), "val_n": len(rows)}
    torch.save(state, args.weights)
    print("stored", state["ensemble"], "in", args.weights)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "trocr", "fit", "table"])
    ap.add_argument("--sources", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--trocr", type=int, default=64, help="TrOCR: samples per source")
    ap.add_argument("--out", default=str(ROOT / "data/text_general/bench.json"))
    ap.add_argument("--weights", default=str(ROOT / "weights/general_ocr.pt"))
    ap.add_argument("--val-per-source", type=int, default=200)
    ap.add_argument("--per-source", type=int, default=200)
    ap.add_argument("--threads", type=int, default=2, help="torch threads for `run`")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args)
    elif args.cmd == "trocr":
        run_trocr(args)
    elif args.cmd == "fit":
        fit(args)
    else:
        results = json.loads(Path(args.out).read_text())
        tp = Path(_trocr_path(args.out))
        if tp.exists():
            for k, v in json.loads(tp.read_text()).items():
                if k in results:
                    results[k].update(v)
        print(table(results))


if __name__ == "__main__":
    main()
