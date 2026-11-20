"""
Local fixture server for the extension's end-to-end test. Never talks to the internet.

    /fixtures/*                  static fixture pages (extension/test/fixtures)
    /aws/problem?sid=&mode=      mocked AWS WAF captcha /problem (data/samples/problem_visual.json)
    /aws/verify?sid=             mocked /verify: answer [0,2,4,6,8] wins; mode=wrongfirst
                                 rejects the first answer and serves a new puzzle
    /bls/grid?grid=&variant=     BLS-like grid page: 9 real tiles + hidden/covered/off-screen decoys
    /bls/submit                  BLS form target (logs the selected tile ids)
    /blspl/login?img=            BLS Poland-like login page with a text captcha
    /blspl/captcha.jpg?img=      the captcha image (every download is logged)
    /mockapi/*                   CapSolver-compatible mock (ImageToTextTask) for when the
                                 real service has no model for it
    /_log                        everything above that the test asserts on
"""

import argparse
import base64
import html
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
ROOT = HERE.parent.parent
AWS_SAMPLE = ROOT / "data" / "samples" / "problem_visual.json"
AWS_ANSWER = [0, 2, 4, 6, 8]
BLS_REAL = Path("/home/gg/projects/bls_capsolver/data/real")
BLS_PL = Path("/home/gg/projects/BlsPolandBott_dev/temp")
MOCK_TEXT_MIXED = "aB3dE"   # generic module: upper/lower case must reach the input unchanged
MOCK_TEXT = "40719"

LOG = {"verify": [], "bls_submit": [], "blspl_images": [], "mockapi": [], "blspl_submit": []}
SESSIONS = {}
TILE_IDS = {}  # tile element id -> "real<i>" / "decoy"
LOCK = threading.Lock()


def load_bls_grids():
    grids = {}
    with open(BLS_REAL / "grids.jsonl") as fh:
        for line in fh:
            g = json.loads(line)
            grids[g["grid"]] = g
    labels = json.loads((BLS_REAL / "labels.json").read_text())
    return grids, labels


GRIDS, LABELS = load_bls_grids()


def bls_truth(grid_id):
    g = GRIDS[grid_id]
    return [i for i, t in enumerate(g["tiles"]) if LABELS.get(t) == g["target"]]


def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def bls_page(grid_id, variant):
    """A page shaped like the BLS captcha: decoy labels/tiles hidden in several ways."""
    g = GRIDS[grid_id]
    rnd = random.Random(grid_id)
    others = [k for k in GRIDS if k != grid_id]
    decoy_grid = GRIDS[rnd.choice(others)]
    decoys = [BLS_REAL / "raw" / t for t in decoy_grid["tiles"]]
    target = g["target"]
    fake_targets = [str(rnd.randint(100, 999)) for _ in range(4)]
    labels = [
        f'<div class="box-label" style="display:none">Please select all boxes with number {fake_targets[0]}</div>',
        f'<div class="box-label" style="position:absolute;left:-4000px">Please select all boxes with number {fake_targets[1]}</div>',
        # covered by the real label's container
        f'<div class="lblwrap"><div class="box-label" style="position:absolute;top:0;left:0;z-index:1">'
        f'Please select all boxes with number {fake_targets[2]}</div>'
        f'<div class="box-label" style="position:relative;z-index:5;background:#fff">'
        f'Please select all boxes with number {target}</div></div>',
        f'<div class="box-label" style="visibility:hidden">Please select all boxes with number {fake_targets[3]}</div>',
    ]
    cells = []
    for i, tile in enumerate(g["tiles"]):
        rid = "t%08x" % rnd.getrandbits(32)
        did = "t%08x" % rnd.getrandbits(32)
        with LOCK:
            TILE_IDS[rid], TILE_IDS[did] = i, "decoy"
        real = (f'<img class="captcha-img" id="{rid}" data-fx="real{i}" src="{data_uri(BLS_REAL / "raw" / tile)}"'
                f' style="position:absolute;inset:0;z-index:{10 + rnd.randint(0, 5)}" onclick="pick(this)">')
        decoy = (f'<img class="captcha-img" id="{did}" data-fx="decoy" src="{data_uri(decoys[i])}"'
                 f' style="position:absolute;inset:0;z-index:{rnd.randint(1, 5)}" onclick="pick(this)">')
        # document order of real/decoy varies; z-index decides who is on top
        pair = real + decoy if rnd.random() < 0.5 else decoy + real
        cells.append(f'<div class="cell">{pair}</div>')
    hidden_decoys = "".join(
        f'<img class="captcha-img" data-fx="decoy" src="{data_uri(p)}" style="{style}" onclick="pick(this)">'
        for p, style in zip(decoys[:3], ["display:none", "position:absolute;left:-5000px;top:0",
                                          "opacity:0"]))
    password = ('<label>Password <input type="password" id="pwd" name="pwd"></label>'
                if variant == "login" else "")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>BLS grid fixture</title>
<style>
 body {{ font-family: sans-serif; margin: 20px; }}
 .lblwrap {{ position: relative; height: 24px; margin-bottom: 8px; }}
 .grid {{ display: grid; grid-template-columns: repeat(3, 110px); gap: 6px; }}
 .cell {{ position: relative; width: 110px; height: 110px; }}
 .cell img {{ width: 110px; height: 110px; cursor: pointer; border: 3px solid transparent; box-sizing: border-box; }}
 .cell img.img-selected {{ border-color: #2a7; }}
</style></head><body>
<h1>BLS grid fixture ({variant})</h1>
<form id="captchaForm" method="post" action="/bls/submit?grid={grid_id}&variant={variant}">
 {''.join(labels)}
 <div class="grid">{''.join(cells)}</div>
 {hidden_decoys}
 <input type="hidden" id="SelectedImages" name="SelectedImages">
 {password}
 <button type="submit" id="btnVerify">Verify</button>
</form>
<script>
 function pick(img) {{
   img.classList.toggle('img-selected');
   const ids = [...document.querySelectorAll('img.captcha-img.img-selected')].map(i => i.id);
   document.getElementById('SelectedImages').value = ids.join(',');
 }}
</script></body></html>"""


def blspl_page(img):
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>BLS Poland login fixture</title></head>
<body style="font-family:sans-serif;margin:20px">
<h1>BLS Poland login fixture</h1>
<form id="loginForm" method="post" action="/blspl/submit">
  <label>Email <input type="email" name="email" value="someone@example.com"></label><br><br>
  <img src="/blspl/captcha.jpg?img={html.escape(img)}" alt="captcha" id="captcha_image" width="160" height="55"><br>
  <input type="text" name="captcha_code" id="captcha_code" maxlength="5" autocomplete="off">
  <button type="submit">Login</button>
</form>
<script>
  window.__events = [];
  const inp = document.getElementById('captcha_code');
  for (const t of ['input', 'change']) inp.addEventListener(t, () => window.__events.push(t + ':' + inp.value));
</script></body></html>"""


def generic_page(img):
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Generic text captcha fixture</title></head>
<body style="font-family:sans-serif;margin:20px">
<h1>Generic image captcha fixture</h1>
<form method="post" action="/blspl/submit">
  <img src="/blspl/captcha.jpg?img={html.escape(img)}" id="secImage" class="verify-img" alt="captcha"><br>
  <input type="text" name="verification_code" id="verification_code" placeholder="Enter the code">
  <button type="submit">Send</button>
</form></body></html>"""


GENERIC_VARIANTS = {
    # a label describes the input; the picture has no captcha-like attributes at all
    "label": """<form method="post" action="/blspl/submit">
  <p><img src="/blspl/captcha.jpg?img={img}" id="pic" width="200" height="60"></p>
  <label for="ans">Enter the characters shown in the picture</label>
  <input type="text" name="answer" id="ans"> <button type="submit">Send</button></form>""",
    # the captcha is drawn on a canvas, the input is described by its placeholder only
    "canvas": """<form method="post" action="/blspl/submit">
  <canvas id="cv" width="200" height="60"></canvas><br>
  <input type="text" name="f2" id="sec" placeholder="Security code"> <button type="submit">Send</button></form>
<script>const im = new Image(); im.onload = () => document.getElementById('cv').getContext('2d').drawImage(im, 0, 0, 200, 60);
im.src = '/blspl/captcha.jpg?img={img}';</script>""",
    # everything inside an open shadow root, Turkish wording
    "shadow": """<div id="host"></div><script>
const root = document.getElementById('host').attachShadow({{mode: 'open'}});
root.innerHTML = '<p>Resimdeki kodu giriniz</p><img id="simg" src="/blspl/captcha.jpg?img={img}">' +
  '<input type="text" id="kod" name="guvenlik">';
</script>""",
    # a normal login form with a logo: nothing must be filled
    "negative": """<form method="post" action="/blspl/submit">
  <img src="/static/logo.jpg?img={img}" id="logo" width="200" height="60" alt="Acme">
  <input type="text" name="username" id="user" placeholder="User name">
  <input type="password" name="password" id="pw">
  <input type="text" name="q" id="search" placeholder="Search"> <button type="submit">Log in</button></form>""",
    # the picture comes from another origin (localhost vs 127.0.0.1): read from a screenshot
    "cross": """<form method="post" action="/blspl/submit">
  <img src="http://localhost:{port}/blspl/captcha.jpg?img={img}" id="xpic" alt="captcha">
  <input type="text" name="captcha_text" id="xin"> <button type="submit">Send</button></form>""",
}


# NopeCHA's text-captcha demo layout: many iframes whose src is set by a script, each with
# <img class="captchapict"> and <input class="captcha"> and nothing else to go by
NOPECHA_FRAME = """<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0;font-family:sans-serif"><img class="captchapict" src="/blspl/captcha.jpg?img={img}">
<input class="captcha" type="text"></body></html>"""


def nopecha_like_page(n, imgs):
    frames = "".join("<iframe></iframe>" for _ in range(n))
    srcs = json.dumps([f"/generic/nopecha_frame?img={html.escape(imgs[i % len(imgs)])}&i={i}" for i in range(n)])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>nopecha-like</title>
<style>iframe {{ width: 240px; height: 110px; border: 1px solid #ccc; }}</style></head>
<body><div id="parent">{frames}</div>
<script>const s = {srcs}; document.querySelectorAll('#parent iframe').forEach((f, i) => f.src = s[i]);</script>
</body></html>"""


def generic_variant_page(variant, img, port):
    body = GENERIC_VARIANTS[variant].format(img=html.escape(img), port=port)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>generic {variant}</title></head>
<body style="font-family:sans-serif;margin:20px"><h1>Generic text captcha: {variant}</h1>{body}</body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "fixture/1"

    def log_message(self, *args):  # quiet
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json",
                   {"Access-Control-Allow-Origin": "*"})

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain", {"Access-Control-Allow-Origin": "*",
                                             "Access-Control-Allow-Headers": "*",
                                             "Access-Control-Allow-Methods": "GET, POST"})

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        path = url.path
        if path == "/_log":
            with LOCK:
                return self._json(LOG)
        if path == "/aws/problem":
            problem = json.loads(AWS_SAMPLE.read_text())
            with LOCK:
                SESSIONS.setdefault(q.get("sid", ""), {"mode": q.get("mode", ""), "verifies": 0})
            return self._json(problem)
        if path == "/bls/grid":
            return self._send(200, bls_page(q["grid"], q.get("variant", "appointment")))
        if path == "/blspl/login":
            return self._send(200, blspl_page(q.get("img", "cap-4ot0qvfu3q5.jpg")))
        if path == "/generic/login":
            return self._send(200, generic_page(q.get("img", "cap-br7bzbh49g8.jpg")))
        if path == "/generic/v" and q.get("variant") in GENERIC_VARIANTS:
            return self._send(200, generic_variant_page(q["variant"], q.get("img", "cap-br7bzbh49g8.jpg"),
                                                        self.server.server_address[1]))
        if path == "/generic/nopecha":
            imgs = sorted(p.name for p in BLS_PL.glob("*.jpg"))[:6] or ["cap-br7bzbh49g8.jpg"]
            return self._send(200, nopecha_like_page(int(q.get("n", "12")), imgs))
        if path == "/generic/nopecha_frame":
            return self._send(200, NOPECHA_FRAME.format(img=html.escape(Path(q.get("img", "")).name)))
        if path == "/static/logo.jpg":
            return self._send(200, (BLS_PL / Path(q.get("img", "cap-br7bzbh49g8.jpg")).name).read_bytes(),
                              "image/jpeg")
        if path == "/blspl/captcha.jpg":
            name = Path(q.get("img", "")).name
            with LOCK:
                LOG["blspl_images"].append(name)
            return self._send(200, (BLS_PL / name).read_bytes(), "image/jpeg")
        if path.startswith("/fixtures/"):
            f = (FIXTURES / path[len("/fixtures/"):]).resolve()
            if FIXTURES in f.parents and f.is_file():
                ctype = {"html": "text/html; charset=utf-8", "js": "text/javascript"}.get(
                    f.suffix.lstrip("."), "application/octet-stream")
                return self._send(200, f.read_bytes(), ctype)
        if path == "/mockapi/health":
            return self._json({"ok": True, "version": "mock"})
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        path = url.path
        raw = self._body()
        if path == "/aws/verify":
            body = json.loads(raw or b"{}")
            sid = q.get("sid", "")
            with LOCK:
                sess = SESSIONS.setdefault(sid, {"mode": "", "verifies": 0})
                sess["verifies"] += 1
                solution = sorted(body.get("client_solution") or [])
                wrong_first = sess["mode"] == "wrongfirst" and sess["verifies"] == 1
                ok = solution == AWS_ANSWER and not wrong_first
                LOG["verify"].append({"sid": sid, "solution": solution, "accepted": ok})
            if ok:
                return self._json({"success": True, "captcha_voucher": "voucher-ok"})
            return self._json({"success": False, "problem": json.loads(AWS_SAMPLE.read_text())})
        if path == "/bls/submit":
            form = {k: v[0] for k, v in parse_qs(raw.decode()).items()}
            with LOCK:
                ids = [x for x in form.get("SelectedImages", "").split(",") if x]
                LOG["bls_submit"].append({"grid": q.get("grid"), "variant": q.get("variant"),
                                          "selected": sorted(TILE_IDS.get(x, "unknown") for x in ids)
                                          if all(isinstance(TILE_IDS.get(x), int) for x in ids)
                                          else [TILE_IDS.get(x, "unknown") for x in ids]})
            return self._send(200, "<h1>submitted</h1>")
        if path == "/blspl/submit":
            form = {k: v[0] for k, v in parse_qs(raw.decode()).items()}
            with LOCK:
                LOG["blspl_submit"].append(form.get("captcha_code", ""))
            return self._send(200, "<h1>submitted</h1>")
        if path.startswith("/mockapi/"):
            body = json.loads(raw or b"{}")
            if path.endswith("/getBalance"):
                return self._json({"errorId": 0, "balance": 25000})
            task = body.get("task") or {}
            with LOCK:
                LOG["mockapi"].append({"type": task.get("type"), "module": task.get("module"),
                                       "case": task.get("case"),
                                       "body_head": (task.get("body") or "")[:16],
                                       "body_len": len(task.get("body") or "")})
            if task.get("type") != "ImageToTextTask":
                return self._json({"errorId": 1, "errorCode": "ERROR_TASK_NOT_SUPPORTED",
                                   "errorDescription": "mock only does ImageToTextTask"})
            answer = MOCK_TEXT_MIXED if task.get("module") == "common" else MOCK_TEXT
            return self._json({"errorId": 0, "status": "ready", "taskId": "mock",
                               "solution": {"text": answer, "confidence": 0.97}})
        return self._send(404, "not found", "text/plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8792)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"fixture server on http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
