"""
Real AWS WAF widget check: AWS's own captcha.js (a local copy, never committed) is served
on 127.0.0.1 with a mocked /problem and /verify, and the extension must press Begin, pick
the tiles and submit by itself. Needs the captchosa service on 127.0.0.1:8790 and a client
key in CAPTCHOSA_KEY. Skips when the captcha.js copy is missing.

    CAPTCHOSA_KEY=... .venv/bin/python extension/test/real_widget.py
"""
import json, os, sys, tempfile, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from playwright.sync_api import sync_playwright

EXT = Path(__file__).resolve().parents[1]
JS_PATH = Path(os.environ.get("CAPTCHOSA_AWS_JS", Path(__file__).resolve().parents[2] / "data/audio/js/captcha.js"))
if not JS_PATH.exists():
    print(f"SKIP: no local copy of AWS captcha.js at {JS_PATH}")
    sys.exit(0)
JS = JS_PATH.read_bytes()
PROBLEM = (Path(__file__).resolve().parents[2] / "data/samples/problem_visual.json").read_text()
KEY = os.environ.get("CAPTCHOSA_KEY", "")
PORT = 8795
BASE = "/c7/3a/c9"
LOG = []

PAGE = f"""<!doctype html><html><head><meta charset=utf-8>
<script>window.gokuProps={{"key":"k","iv":"i","context":"c"}};</script>
<script src="{BASE}/captcha.js"></script></head>
<body><div id="captcha-container"></div>
<script>window.addEventListener('load',()=>{{
 window.CaptchaScript.renderCaptcha(document.querySelector('#captcha-container'),
   (v,t)=>{{document.title='SOLVED';}}, {{}});
}});</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(200)
        self.send_header("content-type", ctype)
        self.send_header("access-control-allow-origin", "*")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        LOG.append(("GET", self.path))
        if self.path.startswith(BASE + "/captcha.js"):
            return self._send(JS, "application/javascript")
        if self.path.startswith(BASE + "/problem"):
            return self._send(PROBLEM, "application/json")
        if self.path == "/":
            return self._send(PAGE, "text/html")
        self.send_response(404); self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n).decode(errors="replace")
        LOG.append(("POST", self.path, body[-600:]))
        if self.path.startswith(BASE + "/verify"):
            return self._send(json.dumps({"success": True, "captcha_voucher": "V"}), "application/json")
        if self.path.startswith(BASE + "/report"):
            return self._send("{}", "application/json")
        self._send("{}", "application/json")


srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()

with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        tempfile.mkdtemp(prefix="captchosa-real-"),
        headless=True, channel="chromium", viewport={"width": 1000, "height": 900},
        args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"])
    sw = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker")
    sw.evaluate("""async (key) => { const {settings} = await chrome.storage.local.get('settings');
        await chrome.storage.local.set({settings: CAPTCHOSA.normalize(Object.assign(settings||{},
        {serviceUrl:'http://127.0.0.1:8790', clientKey:key, delayMs:200}))}); }""", KEY)
    page = ctx.new_page()
    page.on("console", lambda m: print("console:", m.type, m.text[:200]))
    page.goto(f"http://127.0.0.1:{PORT}/")
    time.sleep(4)
    print("DOM buttons:", page.evaluate("""() => [...document.querySelectorAll('button')].map(b =>
        (b.id||'') + '|' + b.className + '|' + (b.textContent||'').trim().slice(0,20) + '|' +
        Math.round(b.getBoundingClientRect().width)+'x'+Math.round(b.getBoundingClientRect().height))"""))
    page.screenshot(path=str(Path(__file__).parent / "s1.png"))
    begin = None  # the extension must press Begin itself
    if begin:
        print("clicking Begin (as the user)"); begin.click(); time.sleep(8)
    else:
        time.sleep(8)
    page.screenshot(path=str(Path(__file__).parent / "s2.png"))
    posted = [e for e in LOG if e[0] == "POST" and "/verify" in e[1]]
    ok = page.title() == "SOLVED" and posted and '"client_solution":[]' not in posted[-1][2]
    print("PASS" if ok else "FAIL", "- begin pressed + tiles submitted:", posted[-1][2][-160:] if posted else "no /verify")
    ctx.close()
    sys.exit(0 if ok else 1)
