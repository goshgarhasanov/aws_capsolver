"""
Real (and third-party synthetic) labelled captcha datasets for the general text model.

Every source is downloaded from a public place (Hugging Face datasets, one GitHub mirror,
the user's own local samples), normalised and written as a "pack":

    data/text_general/packs/<source>/{train,test}.bin   concatenated encoded images
    data/text_general/packs/<source>/{train,test}.json  {"keys", "labels", "offsets"}

The test split is FROZEN: it is chosen deterministically (official test split when the
dataset has one, else a hash of the sample key) and never used for training, so per-source
accuracy can be reported (tools/general_bench.py). Datasets are never committed.

    .venv/bin/python tools/general_data.py fonts            # OFL fonts for the generator
    .venv/bin/python tools/general_data.py download [names]
    .venv/bin/python tools/general_data.py build [names]
    .venv/bin/python tools/general_data.py list
    .venv/bin/python tools/general_data.py sheet <name>     # contact sheet of the test split
"""

import argparse
import csv
import glob
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "text_general"
RAW = BASE / "raw"
PACKS = BASE / "packs"
LABEL_RE = re.compile(r"^[0-9A-Za-z]{1,12}$")

Sample = Tuple[str, str, bytes, Optional[str]]   # key, label, image bytes, split hint


@dataclass
class Source:
    name: str
    origin: str                    # where it comes from (HF id / URL / local path)
    license: str
    kind: str                      # "real" (collected from a live site / product) or "synthetic"
    style: str                     # one-line description of the look
    case_sensitive: bool           # do the labels distinguish upper / lower case reliably?
    loader: Callable[[], Iterator[Sample]]
    hf: Optional[Tuple[str, List[str]]] = None    # (dataset id, allow_patterns) to download
    n_test: int = 500
    train_cap: int = 60000
    eval_only: bool = False
    charset: Optional[str] = None  # the hint a client would realistically send
    lengths: Tuple[int, int] = (0, 0)
    fold_train: bool = False       # labels do not follow the glyph case: train case-folded


def _raw(hf_id: str) -> Path:
    return RAW / hf_id.replace("/", "__")


def _parquet_rows(pattern: str, columns: List[str]) -> Iterator[dict]:
    import pyarrow.parquet as pq
    for path in sorted(glob.glob(pattern)):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2048, columns=columns):
            yield from batch.to_pylist()


def _img_bytes(value) -> bytes:
    if isinstance(value, dict):
        return value["bytes"]
    return value


# loaders -----------------------------------------------------------------------
def load_captcha_images_v2():
    with zipfile.ZipFile(RAW / "captcha_images_v2.zip") as zf:
        for name in sorted(zf.namelist()):
            if name.lower().endswith((".png", ".jpg")) and "__MACOSX" not in name:
                yield name, Path(name).stem, zf.read(name), None


def load_project_sloth():
    for split in ("train", "validation", "test"):
        with tarfile.open(_raw("project-sloth/captcha-images") / "data" / f"{split}.tar.gz") as tf:
            for m in tf:
                if m.isfile() and m.name.lower().endswith((".jpg", ".png")):
                    yield (m.name, Path(m.name).name.split(".")[0], tf.extractfile(m).read(),
                           "test" if split == "test" else "train")


def load_recaptcha_v1():
    root = _raw("swargaraj/recaptcha-v1-captchas")
    for path in sorted(root.glob("train/**/*.png")):
        yield str(path.relative_to(root)), path.stem, path.read_bytes(), None


def load_simplecaptcha():
    root = _raw("muddokon/outlined-text-captcha")
    for split in ("train", "validation", "test"):
        with zipfile.ZipFile(root / f"{split}.zip") as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith((".jpg", ".png")):
                    yield f"{split}/{name}", Path(name).stem, zf.read(name), None


def load_gib():
    root = _raw("Tuna0000/gib-captcha-labeled")
    for split in ("train", "validation", "test"):
        for i, r in enumerate(_parquet_rows(str(root / "data" / f"{split}-*.parquet"),
                                            ["image", "text"])):
            yield f"{split}/{i}", r["text"], _img_bytes(r["image"]), \
                "test" if split == "test" else "train"


def load_ekyb():
    root = _raw("dathuynh1108/captcha-to-text")
    for split in ("train", "validation"):
        meta = root / split / "metadata.csv"
        if not meta.exists():
            continue
        with open(meta, newline="") as fh:
            for row in csv.DictReader(fh):
                path = root / split / row["file_name"]
                if path.exists():
                    yield (f"{split}/{row['file_name']}", row["text"], path.read_bytes(),
                           "test" if split == "validation" else "train")


def load_hansda():
    root = _raw("Hansda/captcha-dataset")
    for split in ("train", "test"):
        for i, r in enumerate(_parquet_rows(str(root / "data" / f"{split}-*.parquet"),
                                            ["image", "text"])):
            yield f"{split}/{i}", r["text"], _img_bytes(r["image"]), split


def load_lumasik():
    for i, r in enumerate(_parquet_rows(str(_raw("lumasik/captcha-25k") / "train.parquet"),
                                        ["image", "label"])):
        yield str(i), r["label"], _img_bytes(r["image"]), None


def load_captchaocr500k():
    for r in _parquet_rows(str(_raw("AvinashRicky/CaptchaOCR-500K") / "data" / "test-*.parquet"),
                           ["index", "text", "image"]):
        yield str(r["index"]), r["text"], _img_bytes(r["image"]), None


def load_ormeshein():
    with zipfile.ZipFile(_raw("ormeshein/captcha") / "captchas.zip") as zf:
        names = set(zf.namelist())
        rows = list(csv.DictReader(io.TextIOWrapper(zf.open("labels.csv"), "utf-8")))
        for row in rows:
            name = os.path.basename(row["filename"])
            if name in names:
                yield name, row["label"], zf.read(name), None


def load_botdetect():
    path = _raw("squirelmail/dataset-BotDetect-CAPTCHA-Generator") / "data" / "dataset_1000_v2.tar.gz"
    with tarfile.open(path) as tf:
        for m in tf:
            if m.isfile() and m.name.lower().endswith(".png"):
                parts = Path(m.name).parts
                style = next((p for p in parts if p.startswith("style")), "style?")
                yield f"{style}/{parts[-1]}", Path(m.name).stem.split("_")[0], \
                    tf.extractfile(m).read(), None


def load_nj_biergarten():
    for r in _parquet_rows(str(_raw("phunc20/nj_biergarten_captcha") / "data" / "train-00000-*.parquet"),
                           ["__key__", "jpg"]):
        key = r["__key__"]
        yield key, key.rsplit("_", 1)[-1], _img_bytes(r["jpg"]), None


def load_yusuf():
    for i, r in enumerate(_parquet_rows(str(_raw("yusuf802/captcha_dataset") / "data" / "*.parquet"),
                                        ["image", "label"])):
        yield str(i), r["label"], _img_bytes(r["image"]), None


def load_cybertruck():
    for i, r in enumerate(_parquet_rows(str(_raw("cybertruck32489/captcha_50k") / "data" / "*.parquet"),
                                        ["image", "solve"])):
        yield str(i), r["solve"], _img_bytes(r["image"]), None


def load_local():
    """The user's own captchas (labels.json next to the images); evaluation only."""
    for d in sorted(glob.glob(str(BASE / "local" / "*"))):
        labels = json.loads(Path(d, "labels.json").read_text())
        for name, label in sorted(labels.items()):
            p = Path(d, name)
            if not name.startswith("_") and p.exists():
                yield f"{Path(d).name}/{name}", label, p.read_bytes(), "test"


def load_pycaptcha():
    """python `captcha` ImageCaptcha with its bundled default font (synthetic, held out)."""
    import random as _random
    from captcha.image import ImageCaptcha
    rng = _random.Random(1234)
    gen = ImageCaptcha()
    pool = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    _random.seed(99)
    for i in range(1500):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(4, 6)))
        buf = io.BytesIO()
        gen.generate_image(text).save(buf, format="PNG")
        yield str(i), text, buf.getvalue(), "test"


SOURCES: Dict[str, Source] = {s.name: s for s in [
    Source("captcha_images_v2", "https://github.com/AakashKumarNain/CaptchaCracker "
           "(captcha_images_v2.zip, the Kaggle 'captcha-version-2-images' set by R. Wilhelmy)",
           "unspecified (public Kaggle dataset; research use)", "real",
           "5 lower-case/digit chars, grey, noise line + blur", False, load_captcha_images_v2,
           n_test=300, charset="alnum", lengths=(5, 5)),
    Source("project_sloth", "hf:project-sloth/captcha-images", "WTFPL", "synthetic",
           "6 digits, python-captcha-like", False, load_project_sloth,
           hf=("project-sloth/captcha-images", ["data/*"]), charset="digits", lengths=(6, 6)),
    Source("recaptcha_v1", "hf:swargaraj/recaptcha-v1-captchas",
           "other (see dataset LICENSE.txt; research use)", "real",
           "reCAPTCHA v1 words, 8-10 lower-case letters, warped", False, load_recaptcha_v1,
           hf=("swargaraj/recaptcha-v1-captchas", ["*"]), n_test=1000, charset="letters",
           lengths=(8, 10), fold_train=True),
    Source("simplecaptcha", "hf:muddokon/outlined-text-captcha", "MIT", "real",
           "SimpleCaptcha (Java) outlined glyphs on gradient, 5 chars", False,
           load_simplecaptcha, hf=("muddokon/outlined-text-captcha", ["*.zip"]),
           n_test=300, lengths=(5, 5)),
    Source("gib_tr", "hf:Tuna0000/gib-captcha-labeled", "Apache-2.0", "real",
           "Turkish revenue administration (GIB), 6 upper-case/digit chars", False, load_gib,
           hf=("Tuna0000/gib-captcha-labeled", ["data/*"]), lengths=(6, 6)),
    Source("ekyb_vn", "hf:dathuynh1108/captcha-to-text", "unspecified (public HF dataset)",
           "real", "Vietnamese government portals (two sites)", False, load_ekyb,
           hf=("dathuynh1108/captcha-to-text", ["*"])),
    Source("hansda", "hf:Hansda/captcha-dataset", "Apache-2.0", "synthetic",
           "distorted per-char fonts, warps, lines, 4-7 chars", False, load_hansda,
           hf=("Hansda/captcha-dataset", ["data/*"]), fold_train=True),
    Source("lumasik", "hf:lumasik/captcha-25k", "MIT", "synthetic",
           "12+ noise filters (blur, salt/pepper, ink bleed), upper/digits", False,
           load_lumasik, hf=("lumasik/captcha-25k", ["*.parquet"])),
    Source("captchaocr500k", "hf:AvinashRicky/CaptchaOCR-500K (test split only)", "MIT",
           "synthetic", "python-captcha-like, labels lower-cased (glyphs are not)", False,
           load_captchaocr500k, hf=("AvinashRicky/CaptchaOCR-500K", ["data/test-*"]),
           fold_train=True),
    Source("ormeshein", "hf:ormeshein/captcha", "MIT", "synthetic",
           "python-captcha ImageCaptcha look, 4 upper/digit chars", False, load_ormeshein,
           hf=("ormeshein/captcha", ["*.zip"]), n_test=300),
    Source("botdetect", "hf:squirelmail/dataset-BotDetect-CAPTCHA-Generator (dataset_1000_v2)",
           "Apache-2.0", "synthetic", "BotDetect CAPTCHA, 60 styles, 5 upper/digit chars",
           False, load_botdetect,
           hf=("squirelmail/dataset-BotDetect-CAPTCHA-Generator", ["data/dataset_1000_v2.tar.gz"]),
           n_test=1200),
    Source("nj_biergarten", "hf:phunc20/nj_biergarten_captcha (shard 0 of 3)", "GPL-3.0",
           "real", "crawled site captcha, 6 chars [a-z1-9]", False, load_nj_biergarten,
           hf=("phunc20/nj_biergarten_captcha", ["data/train-00000-*"]), n_test=1000),
    Source("yusuf802", "hf:yusuf802/captcha_dataset (shard 0)", "unspecified (public HF dataset)", "synthetic",
           "coloured / light-on-dark, curves, mixed case, 5 chars", True, load_yusuf, hf=("yusuf802/captcha_dataset", ["data/train-00000-*"])),
    Source("cybertruck", "hf:cybertruck32489/captcha_50k (shard 0)", "unspecified (public HF dataset)",
           "synthetic", "strike lines on textured grey, varied fonts, 5 lower/digit", False,
           load_cybertruck, hf=("cybertruck32489/captcha_50k", ["data/train-00000-*"])),
    Source("pycaptcha_default", "generated: python `captcha` 0.7.1 ImageCaptcha, default font",
           "BSD (library); images generated locally", "synthetic",
           "python-captcha default look, 4-6 mixed-case chars", True, load_pycaptcha,
           eval_only=True, n_test=1500),
    Source("local", "the user's own samples (CaptchaBypasser/samples, BlsPolandBott*/temp)",
           "private, never committed", "real", "live site captchas (labelled by eye)", False,
           load_local, eval_only=True, n_test=10 ** 6),
]}


# packs -------------------------------------------------------------------------
def _h(key: str) -> float:
    return int(hashlib.sha1(key.encode()).hexdigest()[:12], 16) / 16 ** 12


def write_pack(path: Path, items: List[Tuple[str, str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    with open(str(path) + ".bin", "wb") as fh:
        for _, _, data in items:
            fh.write(data)
            offsets.append(offsets[-1] + len(data))
    Path(str(path) + ".json").write_text(json.dumps({
        "keys": [k for k, _, _ in items], "labels": [l for _, l, _ in items],
        "offsets": offsets}))


def read_pack(source: str, split: str):
    """(labels, list of image bytes) of a pack; FileNotFoundError when not built."""
    base = PACKS / source / split
    meta = json.loads(Path(str(base) + ".json").read_text())
    blob = Path(str(base) + ".bin").read_bytes()
    off = meta["offsets"]
    return meta["keys"], meta["labels"], [blob[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def build(src: Source) -> Dict[str, int]:
    from PIL import Image
    items, bad = [], 0
    for key, label, data, hint in src.loader():
        label = (label or "").strip()
        if not LABEL_RE.match(label):
            bad += 1
            continue
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
        except Exception:
            bad += 1
            continue
        items.append((key, label, data, hint))
    has_hint = any(h == "test" for *_, h in items)
    pool = [it for it in items if it[3] == "test"] if has_hint else items
    pool = sorted(pool, key=lambda it: _h(it[0]))
    test = pool[:src.n_test]
    test_keys = {it[0] for it in test}
    train = [] if src.eval_only else [it for it in items if it[0] not in test_keys]
    train = sorted(train, key=lambda it: _h("t" + it[0]))[:src.train_cap]
    write_pack(PACKS / src.name / "test", [(k, l, d) for k, l, d, _ in test])
    if train:
        write_pack(PACKS / src.name / "train", [(k, l, d) for k, l, d, _ in train])
    stats = {"total": len(items), "skipped": bad, "test": len(test), "train": len(train)}
    Path(PACKS / src.name / "stats.json").write_text(json.dumps(stats))
    return stats


def download(src: Source) -> None:
    if src.name == "captcha_images_v2":
        RAW.mkdir(parents=True, exist_ok=True)
        dst = RAW / "captcha_images_v2.zip"
        if not dst.exists():
            urllib.request.urlretrieve("https://github.com/AakashKumarNain/CaptchaCracker/raw/"
                                       "master/captcha_images_v2.zip", dst)
        return
    if src.hf is None:
        return
    from huggingface_hub import snapshot_download
    snapshot_download(src.hf[0], repo_type="dataset", allow_patterns=src.hf[1],
                      local_dir=str(_raw(src.hf[0])), max_workers=8)


FONT_FAMILIES = (
    "opensans montserrat oswald merriweather playfairdisplay lobster pacifico bangers comicneue "
    "indieflower permanentmarker specialelite courierprime anton bebasneue righteous fredoka "
    "kalam caveat shadowsintolight patrickhand amaticsc abrilfatface alfaslabone arvo bitter "
    "cabin creteround dosis exo2 josefinsans kanit nunito orbitron oxygen ptsans ptserif "
    "quicksand raleway rubik teko titilliumweb varelaround zillaslab blackopsone pressstart2p "
    "vt323 sharetechmono sourcecodepro inconsolata spacemono gloriahallelujah rocksalt creepster "
    "bungee chewy luckiestguy russoone staatliches archivoblack passionone sigmarone bowlbyone "
    "architectsdaughter handlee coveredbyyourgrace satisfy greatvibes dancingscript sacramento "
    "cinzel lora notable monoton audiowide changaone ultra rye specialgothic hennypenny frijole"
).split()


def fonts() -> None:
    """OFL fonts from github.com/google/fonts (non-italic .ttf files, two per family)."""
    dst = BASE / "fonts"
    dst.mkdir(parents=True, exist_ok=True)
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    for fam in FONT_FAMILIES:
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/google/fonts/contents/ofl/{fam}", headers=headers)
            entries = json.loads(urllib.request.urlopen(req, timeout=30).read())
        except Exception as exc:  # missing family / rate limit: skip it
            print(f"{fam}: {exc}", file=sys.stderr)
            continue
        ttfs = [e for e in entries if e["name"].endswith(".ttf") and "Italic" not in e["name"]]
        for e in ttfs[:2]:
            out = dst / e["name"]
            if not out.exists():
                urllib.request.urlretrieve(e["download_url"], out)
    cache = BASE / "fonts.json"
    if cache.exists():
        cache.unlink()  # the generator re-validates the font list


def sheet(name: str, out: str, n: int = 40) -> None:
    from PIL import Image, ImageDraw
    _, labels, images = read_pack(name, "test")
    cols, cw, ch = 4, 256, 84
    rows = (min(n, len(labels)) + cols - 1) // cols
    img = Image.new("RGB", (cols * cw, rows * ch), "white")
    d = ImageDraw.Draw(img)
    for i, (label, data) in enumerate(zip(labels[:n], images[:n])):
        x, y = (i % cols) * cw, (i // cols) * ch
        with Image.open(io.BytesIO(data)) as im:
            img.paste(im.convert("RGB").resize((cw - 4, 64)), (x + 2, y))
        d.text((x + 4, y + 66), label, fill="red")
    img.save(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["fonts", "download", "build", "list", "sheet"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--out", default="sheet.png")
    args = ap.parse_args(argv)
    names = args.names or list(SOURCES)
    if args.cmd == "fonts":
        fonts()
    elif args.cmd == "list":
        for s in SOURCES.values():
            st = PACKS / s.name / "stats.json"
            print(f"{s.name:20s} {s.kind:9s} {s.license:40s} "
                  f"{st.read_text() if st.exists() else '(not built)'}")
    elif args.cmd == "sheet":
        sheet(names[0], args.out)
    else:
        for name in names:
            src = SOURCES[name]
            try:
                if args.cmd == "download":
                    download(src)
                    print(f"{name}: downloaded")
                else:
                    print(f"{name}: {build(src)}")
            except (OSError, KeyError, ValueError) as exc:
                print(f"{name}: FAILED {exc.__class__.__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
