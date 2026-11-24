"""
Live reCAPTCHA v2 check on a public solver demo page (default: 2captcha.com/demo/recaptcha-v2).

An unpacked copy of the extension is loaded into a fresh Chromium profile (no cookies, no
accounts). The extension talks to a small *recording proxy* started here, which forwards every
request unchanged to the real captchosa service (default http://127.0.0.1:8790) and saves each
ReCaptchaV2Classification task it sees (payload picture, question, size, model answer). The
harness watches the challenge frame from outside (Playwright) and labels each saved task with
what reCAPTCHA did next (accepted / rejected / new images / reloaded). Nothing is added to the
extension itself.

Output (git-ignored):
    data/recaptcha/live/img/<sha1>.<ext>    the picture exactly as the extension sent it
    data/recaptcha/live/index.jsonl         one line per classified picture
    data/recaptcha/live/runs.jsonl          one line per attempt (token, rounds, time, reason)
    data/recaptcha/live/dom/                bframe HTML snapshots (--dump-dom)

Be polite: attempts are paced (--pause, >= 20 s) and capped (--attempts, <= 40). The run
stops at once when Google shows its "automated queries" / "try again later" block.

    CAPTCHOSA_KEY=CX-... .venv/bin/python extension/test/live_recaptcha.py --attempts 5
"""

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

import websocket  # websocket-client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
REPO = EXT.parent
OUT = REPO / "data" / "recaptcha" / "live"
DEMO = "https://2captcha.com/demo/recaptcha-v2"
TASK_TYPE = "ReCaptchaV2Classification"

_lock = threading.Lock()
TASKS = []        # recorded ReCaptchaV2Classification calls of the current attempt


def now():
    return time.time()


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def client_key():
    key = os.environ.get("CAPTCHOSA_KEY", "")
    if key:
        return key
    try:
        keys = json.loads((REPO / "keys.json").read_text())
        return next(k for k in keys if k.startswith("CX-"))
    except Exception:
        return ""


# ------------------------------------------------------------------ recording proxy

def image_ext(raw):
    if raw[:3] == b"\xff\xd8\xff":
        return "jpg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    return "bin"


def make_proxy(upstream):
    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _forward(self, method, body):
            req = urllib.request.Request(upstream + self.path, data=body, method=method,
                                         headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.status, r.read(), r.headers.get("Content-Type", "application/json")
            except urllib.error.HTTPError as e:
                return e.code, e.read(), e.headers.get("Content-Type", "application/json")
            except Exception as e:  # upstream down: answer like the service would
                return 502, json.dumps({"errorId": 1, "errorCode": "ERROR_PROXY",
                                        "errorDescription": str(e)}).encode(), "application/json"

        def _reply(self, status, data, ctype):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()

        def do_GET(self):
            self._reply(*self._forward("GET", None))

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            t0 = now()
            status, data, ctype = self._forward("POST", body)
            try:
                record(self.path, body, data, t0)
            except Exception as e:
                log(f"proxy: record failed: {e}")
            self._reply(status, data, ctype)

    return Proxy


def record(path, body, data, t0):
    if not path.startswith("/createTask"):
        return
    req = json.loads(body or b"{}")
    task = req.get("task") or {}
    if task.get("type") != TASK_TYPE:
        return
    raw = base64.b64decode(task.get("image") or "")
    sha = hashlib.sha1(raw).hexdigest()
    name = f"{sha}.{image_ext(raw)}"
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    p = OUT / "img" / name
    if not p.exists():
        p.write_bytes(raw)
    resp = json.loads(data or b"{}")
    entry = {"t": t0, "ms": int((now() - t0) * 1000), "image": f"img/{name}", "sha1": sha,
             "question": task.get("question"), "size": task.get("size")}
    if resp.get("errorId"):
        entry["answer"] = {"error": resp.get("errorCode"), "message": resp.get("errorDescription")}
    else:
        entry["answer"] = resp.get("solution")
    with _lock:
        TASKS.append(entry)
    log(f"  task size={entry['size']} q={entry['question']!r} -> {json.dumps(entry['answer'])[:120]}")


# ------------------------------------------------------------------ frame watching

SNAP_JS = r"""() => {
  const vis = (el) => { if (!el) return false; const s = getComputedStyle(el);
    if (s.display === 'none' || s.visibility === 'hidden') return false;
    const r = el.getBoundingClientRect(); return r.width > 2 && r.height > 2; };
  const root = document.getElementById('rc-imageselect');
  const table = document.querySelector("table[class*='rc-imageselect-table-']");
  const img = table && table.querySelector("img[class*='rc-image-tile-']:not(.rc-image-tile-11)");
  const desc = document.querySelector('.rc-imageselect-desc-no-canonical, .rc-imageselect-desc');
  const errs = ['rc-imageselect-incorrect-response', 'rc-imageselect-error-select-more',
    'rc-imageselect-error-dynamic-more', 'rc-imageselect-error-select-something']
    .filter((c) => vis(document.querySelector('.' + c)));
  const dos = document.querySelector('.rc-doscaptcha-header, .rc-doscaptcha-body');
  const audio = document.querySelector('.rc-audiochallenge-tdownload, #audio-source');
  return {
    open: vis(root), table: table ? table.className : '', src: img ? img.src : '',
    desc: desc ? desc.innerText.replace(/\s+/g, ' ').trim() : '',
    strong: desc && desc.querySelector('strong') ? desc.querySelector('strong').textContent.trim() : '',
    errors: errs, tiles11: table ? table.querySelectorAll('img.rc-image-tile-11').length : 0,
    selected: table ? [...table.querySelectorAll('td')].map((td, i) => td.classList.contains('rc-imageselect-tileselected') ? i : -1).filter((i) => i >= 0) : [],
    dos: dos && vis(dos) ? dos.innerText.trim().slice(0, 200) : '',
    audio: !!(audio && vis(audio)),
    verifyText: (document.getElementById('recaptcha-verify-button') || {}).textContent || '',
  };
}"""

ANCHOR_JS = r"""() => {
  const b = document.getElementById('recaptcha-anchor');
  const e = document.querySelector('.rc-anchor-error-msg');
  return { checked: b ? b.getAttribute('aria-checked') : null, cls: b ? b.className : '',
           error: e && e.offsetParent ? e.innerText.trim() : '' };
}"""

BADGE_JS = r"""() => {
  const h = document.querySelector('captchosa-badge');
  if (!h || !h.shadowRoot) return null;
  const p = h.shadowRoot.querySelector('.pill');
  return p ? [p.className, p.innerText.trim()] : null;
}"""

BLOCK_WORDS = ("automated queries", "try again later", "unusual traffic", "computer or network may be sending")


class CdpFrame:
    """A cross-origin iframe reached over its own DevTools target (Playwright does not list
    these out-of-process frames on this build). Read-only: only Runtime.evaluate of the
    snapshot functions above."""

    def __init__(self, target):
        self.url = target["url"]
        self.id = target["id"]
        self.ws = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=5, suppress_origin=True)
        self.seq = 0

    def evaluate(self, js):
        self.seq += 1
        self.ws.send(json.dumps({"id": self.seq, "method": "Runtime.evaluate",
                                 "params": {"expression": f"({js})()", "returnByValue": True, "awaitPromise": True}}))
        end = now() + 5
        while now() < end:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.seq:
                res = msg.get("result") or {}
                if "exceptionDetails" in res:
                    raise RuntimeError(json.dumps(res["exceptionDetails"])[:300])
                if "error" in msg:
                    raise RuntimeError(json.dumps(msg["error"])[:300])
                return (res.get("result") or {}).get("value")
        raise TimeoutError("no answer from the frame target")

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


class FrameFinder:
    def __init__(self, port):
        self.port = port
        self.cache = {}

    def find(self):
        try:
            targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/list", timeout=3).read())
        except Exception:
            return None, None
        live = {t["id"] for t in targets}
        for tid in list(self.cache):
            if tid not in live:
                self.cache.pop(tid).close()
        b = a = None
        for t in targets:
            if t.get("type") != "iframe" or not t.get("webSocketDebuggerUrl") and t["id"] not in self.cache:
                continue
            kind = "b" if "/bframe" in t["url"] else "a" if "/anchor" in t["url"] else None
            if not kind or "/recaptcha/" not in t["url"]:
                continue
            if t["id"] not in self.cache:
                try:
                    self.cache[t["id"]] = CdpFrame(t)
                except Exception as e:
                    log(f"  cannot attach to {t['url'][:60]}: {e}")
                    continue
            if kind == "b":
                b = self.cache[t["id"]]
            else:
                a = self.cache[t["id"]]
        return b, a

    def close(self):
        for f in self.cache.values():
            f.close()
        self.cache.clear()


_eval_errors = set()


def safe_eval(frame, js):
    if frame is None:
        return None
    try:
        return frame.evaluate(js)
    except Exception as e:
        msg = str(e).splitlines()[0][:160]
        if msg not in _eval_errors:
            _eval_errors.add(msg)
            log(f"  eval error in {frame.url[:80]}: {msg}")
        return None


def token_of(page):
    try:
        return page.evaluate("() => { const t = document.querySelector('textarea[name=\"g-recaptcha-response\"], #g-recaptcha-response'); return t ? t.value : ''; }")
    except Exception:
        return ""


def attempt(context, finder, url, timeout, dump_dom, n):
    """One page load. Returns the run summary and the list of challenge observations."""
    with _lock:
        TASKS.clear()
    page = context.new_page()
    t0 = now()
    events = []          # (t, snapshot) whenever the frame state changes
    last_key = None
    result = {"attempt": n, "url": url, "start": t0, "token": False, "reason": ""}
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        dumped = set()
        while now() - t0 < timeout:
            time.sleep(0.25)
            b, a = finder.find()
            snap = safe_eval(b, SNAP_JS) or {}
            anc = safe_eval(a, ANCHOR_JS) or {}
            badge = safe_eval(b, BADGE_JS)
            key = json.dumps([snap.get("open"), snap.get("src"), snap.get("desc"), snap.get("errors"), snap.get("tiles11"),
                              snap.get("dos"), anc.get("checked"), anc.get("error"), (badge or [""])[0]])
            if key != last_key:
                last_key = key
                events.append({"t": now(), "snap": snap, "anchor": anc, "badge": badge})
                if snap.get("open") or anc.get("checked") == "true":
                    log(f"  frame: open={snap.get('open')} {snap.get('table')} q={snap.get('desc')!r} errors={snap.get('errors')} "
                        f"tiles11={snap.get('tiles11')} anchor={anc.get('checked')} badge={badge}")
            if dump_dom and b is not None and snap.get("src") and snap["src"] not in dumped:
                dumped.add(snap["src"])
                try:
                    html = b.evaluate("() => document.documentElement.outerHTML")
                    (OUT / "dom").mkdir(parents=True, exist_ok=True)
                    (OUT / "dom" / f"{int(now())}_{len(dumped)}.html").write_text(html)
                except Exception:
                    pass
            text = " ".join([snap.get("dos") or "", anc.get("error") or ""]).lower()
            if snap.get("dos") or any(w in text for w in BLOCK_WORDS):
                result["reason"] = "BLOCKED: " + (snap.get("dos") or anc.get("error"))
                result["blocked"] = True
                break
            if anc.get("checked") == "true":
                time.sleep(1.0)
                result["token"] = bool(token_of(page))
                result["reason"] = "solved" if result["token"] else "checked-without-token"
                break
            if badge and badge[0].endswith("error") and "press Solve" not in badge[1] and "gave up" not in badge[1]:
                result["reason"] = "extension error: " + badge[1]
                break
            if badge and badge[0].endswith("error"):
                result["reason"] = "extension: " + badge[1]
                break
        else:
            result["reason"] = result["reason"] or "timeout"
    except Exception as e:
        result["reason"] = f"harness error: {e}"
    finally:
        result["seconds"] = round(now() - t0, 1)
        with _lock:
            tasks = list(TASKS)
        result["tasks"] = len(tasks)
        result["pictures"] = len([t for t in tasks if t.get("size") in (3, 4)])
        if result.get("token"):
            try:
                page.screenshot(path=str(OUT / "last_solved.png"))
            except Exception:
                pass
        page.close()
        finder.close()
    return result, events, tasks


def label_tasks(tasks, events, result):
    """Outcome of each recorded picture, from what the challenge frame did after it."""
    pics = [t for t in tasks if t.get("size") in (3, 4)]
    out = []
    for i, t in enumerate(tasks):
        # the frame state when the extension asked (the picture it was looking at)
        before = [e for e in events if e["t"] <= t["t"]]
        cur = before[-1]["snap"] if before else {}
        t = dict(t, src=cur.get("src", ""), instruction=cur.get("desc", ""), table=cur.get("table", ""),
                 attempt=result["attempt"])
        if t.get("size") == 1:
            t["outcome"] = "dynamic-tile"
            out.append(t)
            continue
        nxt_pic = next((p for p in pics if p["t"] > t["t"]), None)
        horizon = nxt_pic["t"] if nxt_pic else float("inf")
        after = [e for e in events if t["t"] < e["t"] <= horizon + 0.5]
        outcome = "unknown"
        ans = t.get("answer") or {}
        if "error" in ans:
            outcome = "no-answer"  # the extension reloads (unsupported) or stops (service error)
        if any(e["anchor"].get("checked") == "true" for e in after):
            outcome = "accepted"
        else:
            errs = []
            new_src = None
            for e in after:
                errs += [x for x in e["snap"].get("errors") or [] if x not in errs]
                if e["snap"].get("src") and e["snap"]["src"] != cur.get("src") and new_src is None:
                    new_src = e["snap"]["src"]
            if "error" not in ans:
                if errs:
                    outcome = "rejected:" + ",".join(e.replace("rc-imageselect-", "") for e in errs)
                elif new_src:
                    outcome = "new-images"  # Verify gave another picture without an error line
            elif new_src:
                outcome = "no-answer:reloaded"
        t["outcome"] = outcome
        out.append(t)
    # dynamic tiles inherit the outcome of their picture
    parent = None
    for t in out:
        if t["size"] != 1:
            parent = t
        elif parent:
            t["outcome"] = "dynamic-tile:" + parent["outcome"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEMO)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--pause", type=float, default=30.0, help="seconds between attempts (>= 20)")
    ap.add_argument("--timeout", type=float, default=240.0, help="seconds per attempt")
    ap.add_argument("--upstream", default="http://127.0.0.1:8790")
    ap.add_argument("--proxy-port", type=int, default=8797)
    ap.add_argument("--hl", default="", help="UI language for the widget (?hl= is not on the demo page; "
                    "sets the browser locale instead)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--cdp-port", type=int, default=9339, help="DevTools port used to read the cross-origin frames")
    ap.add_argument("--dump-dom", action="store_true")
    ap.add_argument("--label", default="", help="free text stored with every run line")
    args = ap.parse_args()
    args.attempts = max(1, min(40, args.attempts))
    args.pause = max(20.0, args.pause)
    key = client_key()
    if not key:
        sys.exit("no client key (CAPTCHOSA_KEY or a CX- key in keys.json)")
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        health = json.loads(urllib.request.urlopen(args.upstream + "/health", timeout=5).read())
    except Exception as e:
        sys.exit(f"service not reachable at {args.upstream}: {e}")
    log(f"service: recaptcha={health.get('recaptcha')} version={health.get('version')}")

    srv = ThreadingHTTPServer(("127.0.0.1", args.proxy_port), make_proxy(args.upstream))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    proxy = f"http://127.0.0.1:{args.proxy_port}"

    tmp = Path(tempfile.mkdtemp(prefix="captchosa-live-"))
    ext_dir = tmp / "ext"
    shutil.copytree(EXT, ext_dir, ignore=shutil.ignore_patterns("dist", "test", "docs", "tools"))
    version = json.loads((ext_dir / "manifest.json").read_text())["version"]
    runs = []
    with sync_playwright() as pw:
        launch_args = [f"--disable-extensions-except={ext_dir}", f"--load-extension={ext_dir}",
                       f"--remote-debugging-port={args.cdp_port}"]
        finder = FrameFinder(args.cdp_port)
        if args.hl:
            launch_args.append(f"--lang={args.hl}")
        context = pw.chromium.launch_persistent_context(
            str(tmp / "profile"), headless=args.headless, channel="chromium", args=launch_args,
            locale=args.hl or None, viewport={"width": 1200, "height": 950})
        try:
            sw = context.service_workers[0] if context.service_workers else context.wait_for_event("serviceworker", timeout=15000)
            ext_id = sw.url.split("/")[2]
            time.sleep(1)
            for p in list(context.pages):
                if p.url.startswith("chrome-extension://"):
                    p.close()
            opt = context.new_page()
            opt.goto(f"chrome-extension://{ext_id}/ui/options.html")
            opt.evaluate("""async ([url, key]) => {
                const { settings } = await chrome.storage.local.get('settings');
                const s = CAPTCHOSA.normalize(Object.assign({}, settings || {}, { serviceUrl: url, clientKey: key }));
                await chrome.storage.local.set({ settings: s });
            }""", [proxy, key])
            for n in range(1, args.attempts + 1):
                log(f"attempt {n}/{args.attempts}")
                result, events, tasks = attempt(context, finder, args.url, args.timeout, args.dump_dom, n)
                labelled = label_tasks(tasks, events, result)
                try:
                    ext_log = opt.evaluate("async () => (await chrome.storage.local.get('log')).log || []")
                    result["ext_log"] = [e for e in ext_log if e.get("kind") == "recaptcha" and e.get("ts", 0) / 1000 >= result["start"]]
                    for e in result["ext_log"]:
                        log(f"  extension log: ok={e.get('ok')} {e.get('detail')}")
                except Exception as e:
                    log(f"  extension log unavailable: {e}")
                result["rounds"] = sum(1 for t in labelled if t["size"] != 1)
                result["extension"] = version
                result["label"] = args.label
                result["service"] = {k: health.get(k) for k in ("version", "recaptcha")}
                result["questions"] = [t["question"] for t in labelled if t["size"] != 1]
                result["outcomes"] = [t["outcome"] for t in labelled if t["size"] != 1]
                with open(OUT / "index.jsonl", "a") as fh:
                    for t in labelled:
                        fh.write(json.dumps(t, ensure_ascii=False) + "\n")
                with open(OUT / "runs.jsonl", "a") as fh:
                    fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                runs.append(result)
                log(f"  -> {result['reason']} token={result['token']} {result['seconds']}s pictures={result['pictures']} "
                    f"outcomes={result['outcomes']}")
                if result.get("blocked"):
                    log("Google blocked the session; stopping.")
                    break
                if n < args.attempts:
                    time.sleep(args.pause)
        finally:
            context.close()
            srv.shutdown()
    ok = [r for r in runs if r["token"]]
    log(f"SUMMARY: {len(ok)}/{len(runs)} tokens; "
        f"mean time {sum(r['seconds'] for r in ok) / max(1, len(ok)):.1f}s; "
        f"mean pictures/solve {sum(r['pictures'] for r in ok) / max(1, len(ok)):.1f}")
    reasons = {}
    for r in runs:
        if not r["token"]:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        log(f"  fail x{v}: {k}")


if __name__ == "__main__":
    main()
