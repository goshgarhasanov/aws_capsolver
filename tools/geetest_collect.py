#!/usr/bin/env python3
"""Collect GeeTest v3 / v4 CAPTCHA samples from GeeTest's OWN official public demo pages.

Read-only: the script only loads challenges and downloads their images. It never solves,
submits or verifies anything. Sources:

* v4: https://gt4.geetest.com/ (public demo; captcha_id taken from its JS bundle) -
  ``gcaptcha4.geetest.com/load`` with ``risk_type`` = slide / icon / nine / winlinze / word.
* v3: https://demos.geetest.com/ (public demo) - the page is opened in headless Chromium
  (Playwright), the "click to verify" button is pressed and the ``get.php`` response
  (bg / fullbg / slice or click picture paths) is captured.

Samples land in data/geetest/<type>/<id>/ (gitignored) with a meta.json each.
There is at least ``--delay`` seconds (default 3.5) between two HTTP requests, and the
number of samples per type is capped.

Usage:
    .venv/bin/python tools/geetest_collect.py --types v4_slide,v3_slide --max 200
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import threading
import time
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "geetest"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/130.0.0.0 Safari/537.36")
V4_DEMO_CAPTCHA_ID = "54088bb07d2df3c46b79f80300b0abbe"  # public id of gt4.geetest.com
V4_LOAD = "https://gcaptcha4.geetest.com/load"
STATIC = "https://static.geetest.com/"
V3_DEMO = "https://demos.geetest.com/"
HARD_CAP = 400  # never collect more than this many samples of one type

log = logging.getLogger("geetest_collect")


class Limiter:
    """Global politeness limiter: at least ``delay`` seconds between two requests."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            pause = self._last + self.delay - now
            if pause > 0:
                time.sleep(pause + random.uniform(0, 0.5))
            self._last = time.monotonic()


class Collector:
    def __init__(self, delay: float) -> None:
        self.limiter = Limiter(delay)
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self.icon_cache = DATA / "_static"

    def get(self, url: str, **kw) -> requests.Response:
        self.limiter.wait()
        r = self.s.get(url, timeout=20, **kw)
        r.raise_for_status()
        return r

    def static(self, path: str) -> bytes:
        """Download a static resource once (hint icons repeat a lot -> cached on disk)."""
        path = path.lstrip("/")
        cached = self.icon_cache / path
        if cached.is_file():
            return cached.read_bytes()
        data = self.get(STATIC + path).content
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
        return data

    # ------------------------------------------------------------------ v4
    def v4_load(self, risk_type: str) -> dict:
        params = {"captcha_id": V4_DEMO_CAPTCHA_ID, "challenge": str(uuid.uuid4()),
                  "client_type": "web", "risk_type": risk_type, "lang": "eng",
                  "callback": f"geetest_{int(time.time() * 1000)}"}
        text = self.get(V4_LOAD, params=params, headers={"Referer": "https://gt4.geetest.com/"}).text
        m = re.search(r"^[\w]+\((.*)\)\s*$", text, re.S)
        body = json.loads(m.group(1) if m else text)
        if body.get("status") != "success":
            raise RuntimeError(f"load failed: {body}")
        return body["data"]

    def v4_sample(self, risk_type: str, out: Path) -> None:
        d = self.v4_load(risk_type)
        if d.get("captcha_type") != risk_type:
            raise RuntimeError(f"asked for {risk_type}, got {d.get('captcha_type')}")
        sdir = out / d["lot_number"]
        sdir.mkdir(parents=True, exist_ok=True)
        meta = {k: d.get(k) for k in ("lot_number", "captcha_type", "bg", "slice", "ypos", "imgs",
                                       "ques", "nine_nums")}
        meta["source"] = "gt4.geetest.com demo"
        meta["collected"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if risk_type == "slide":
            (sdir / "bg.png").write_bytes(self.static(d["bg"]))
            (sdir / "slice.png").write_bytes(self.static(d["slice"]))
        elif risk_type in ("icon", "nine", "word"):
            (sdir / "image.jpg").write_bytes(self.static(d["imgs"]))
            for i, q in enumerate(d.get("ques") or []):
                (sdir / f"hint_{i}.png").write_bytes(self.static(q))
        elif risk_type == "winlinze":
            for i, q in enumerate(d.get("imgs") or []):
                (sdir / f"piece_{i}.png").write_bytes(self.static(q))
        (sdir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))

    # ------------------------------------------------------------------ v3
    def v3_samples(self, page: str, out: Path, n: int, want: str) -> int:
        """Open the v3 demo ``page`` n times in headless Chromium; save the captured images."""
        from playwright.sync_api import sync_playwright

        done = 0
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                for _ in range(n * 3):
                    if done >= n:
                        break
                    captured: dict = {}

                    def on_resp(r, captured=captured):
                        if "get.php" in r.url and "is_next=true" in r.url:
                            try:
                                t = r.text()
                                m = re.search(r"^[\w]+\((.*)\)\s*$", t, re.S)
                                captured["data"] = json.loads(m.group(1) if m else t)
                            except Exception as exc:  # noqa: BLE001
                                log.warning("cannot parse get.php: %s", exc)

                    ctx = browser.new_context(user_agent=UA)
                    pg = ctx.new_page()
                    pg.on("response", on_resp)
                    try:
                        self.limiter.wait()
                        pg.goto(V3_DEMO + page, wait_until="networkidle", timeout=30000)
                        pg.wait_for_timeout(1500)
                        pg.click(".geetest_radar_tip", timeout=10000)
                        for _ in range(40):
                            if "data" in captured:
                                break
                            pg.wait_for_timeout(250)
                    except Exception as exc:  # noqa: BLE001 - keep collecting
                        log.warning("v3 %s: %s", page, exc)
                    finally:
                        ctx.close()
                    d = captured.get("data")
                    if not d:
                        continue
                    d = d.get("data", d)
                    try:
                        if want == "slide" and d.get("bg"):
                            sid = Path(d["bg"]).stem + "_" + Path(d["fullbg"]).stem
                            sdir = out / sid
                            sdir.mkdir(parents=True, exist_ok=True)
                            (sdir / "bg.jpg").write_bytes(self.static(d["bg"]))
                            (sdir / "fullbg.jpg").write_bytes(self.static(d["fullbg"]))
                            (sdir / "slice.png").write_bytes(self.static(d["slice"]))
                            meta = {k: d.get(k) for k in ("bg", "fullbg", "slice", "ypos", "xpos", "height")}
                        elif want == "click" and d.get("pic"):
                            sdir = out / Path(d["pic"]).stem
                            sdir.mkdir(parents=True, exist_ok=True)
                            (sdir / "image.jpg").write_bytes(self.static(d["pic"]))
                            meta = {k: d.get(k) for k in ("pic", "pic_type", "spec", "num")}
                        else:
                            log.info("v3 %s: unexpected challenge %s", page, {k: d.get(k) for k in ("type", "pic_type")})
                            continue
                    except requests.RequestException as exc:
                        log.warning("v3 download failed: %s", exc)
                        continue
                    meta["source"] = f"demos.geetest.com/{page}"
                    meta["collected"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    (sdir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
                    done += 1
                    log.info("v3 %s %d/%d", want, done, n)
            finally:
                browser.close()
        return done


V4_TYPES = {"v4_slide": "slide", "v4_icon": "icon", "v4_nine": "nine", "v4_gobang": "winlinze",
            "v4_word": "word"}
V3_TYPES = {"v3_slide": ("slide-float.html", "slide"), "v3_click": ("click-float.html", "click")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", default="v4_slide,v3_slide,v4_icon,v4_nine,v4_gobang,v3_click,v4_word")
    ap.add_argument("--max", type=int, default=200, help="target number of samples per type")
    ap.add_argument("--delay", type=float, default=3.5, help="seconds between requests (>= 3)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    col = Collector(max(3.0, args.delay))
    for t in [t.strip() for t in args.types.split(",") if t.strip()]:
        out = DATA / t
        out.mkdir(parents=True, exist_ok=True)
        have = sum(1 for p in out.iterdir() if (p / "meta.json").is_file())
        need = min(args.max, HARD_CAP) - have
        log.info("%s: have %d, collecting %d more", t, have, max(0, need))
        if need <= 0:
            continue
        if t in V4_TYPES:
            fails = 0
            while need > 0 and fails < 20:
                try:
                    col.v4_sample(V4_TYPES[t], out)
                    need -= 1
                    fails = 0
                except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
                    fails += 1
                    log.warning("%s: %s", t, exc)
                    time.sleep(10)
            log.info("%s done", t)
        elif t in V3_TYPES:
            page, want = V3_TYPES[t]
            col.v3_samples(page, out, need, want)
        else:
            log.error("unknown type %s", t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
