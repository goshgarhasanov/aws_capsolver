"""
Screenshots of the popup and the options page (light + dark) for the README.

Uses the fixture server's mock API (/mockapi) for the connection pill and seeds the
counters/log with sample entries. Run from the repository root:

    .venv/bin/python extension/test/screenshots.py
"""

import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
OUT = EXT / "docs" / "screenshots"
PORT = 8792
MOCK = f"http://127.0.0.1:{PORT}/mockapi"

SEED = """async ({ mock }) => {
  const now = Date.now();
  const min = 60000;
  await chrome.storage.local.set({
    settings: CAPTCHOSA.normalize({ serviceUrl: mock, clientKey: 'demo-key-123', delayMs: 800, retries: 3,
                                    solvers: { geetest: false } }),
    stats: {
      awsGrid: { solved: 128, failed: 3, last: { ok: true, ts: now - 2 * min, detail: 'accepted after 1 round(s)' } },
      awsToken: { solved: 14, failed: 1, last: { ok: true, ts: now - 47 * min, detail: 'aws-waf-token set' } },
      blsGrid: { solved: 342, failed: 6, last: { ok: true, ts: now - 5 * min, detail: '748: picked 1,4,7 (submitted)' } },
      text: { solved: 57, failed: 2, last: { ok: false, ts: now - 3 * 3600000, detail: 'ERROR_CAPTCHA_UNSOLVABLE' } },
    },
    log: [
      { ts: now - 2 * min, kind: 'awsGrid', ok: true, ms: 2140, host: 'shop.example.com', detail: 'accepted after 1 round(s)' },
      { ts: now - 5 * min, kind: 'blsGrid', ok: true, ms: 860, host: 'visa.example.org', detail: '748: picked 1,4,7 (submitted)' },
      { ts: now - 9 * min, kind: 'blsGrid', ok: true, ms: 910, host: 'visa.example.org', detail: '305: picked 0,5' },
      { ts: now - 47 * min, kind: 'awsToken', ok: true, ms: 6420, host: 'app.example.net', detail: 'aws-waf-token set' },
      { ts: now - 3 * 3600000, kind: 'text', ok: false, ms: 410, host: 'login.example.pl', detail: 'ERROR_CAPTCHA_UNSOLVABLE: low confidence' },
    ],
  });
}"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="captchosa-shots-"))
    fixtures = subprocess.Popen([sys.executable, str(HERE / "fixture_server.py"), "--port", str(PORT)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/_log", timeout=1)
                break
            except OSError:
                time.sleep(0.2)
        with sync_playwright() as pw:
            ctx = pw.chromium.launch_persistent_context(
                str(tmp / "profile"), headless=True, channel="chromium", device_scale_factor=2,
                args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}", "--lang=en-US"])
            try:
                sw = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker")
                base = f"chrome-extension://{sw.url.split('/')[2]}"
                time.sleep(1)
                page = ctx.new_page()
                page.goto(f"{base}/ui/options.html")
                page.evaluate(SEED, {"mock": MOCK})
                for scheme in ("light", "dark"):
                    page.emulate_media(color_scheme=scheme)
                    page.set_viewport_size({"width": 360, "height": 600})
                    page.goto(f"{base}/ui/popup.html")
                    page.wait_for_selector("#statusPill.ok", timeout=10000)
                    page.evaluate("document.getElementById('activity').open = true")
                    page.wait_for_timeout(300)
                    page.screenshot(path=str(OUT / f"popup-{scheme}.png"), full_page=True)
                    page.set_viewport_size({"width": 1000, "height": 760})
                    page.goto(f"{base}/ui/options.html")
                    page.wait_for_selector("#statusPill.ok", timeout=10000)
                    page.wait_for_timeout(300)
                    page.screenshot(path=str(OUT / f"options-{scheme}.png"), full_page=True)
                    print("wrote", scheme)
                # the on-page badge in manual mode (Solve button next to the captcha)
                page.evaluate("""async () => { const { settings } = await chrome.storage.local.get('settings');
                    await chrome.storage.local.set({ settings: Object.assign(settings, { autoSolve: false }) }); }""")
                shot = ctx.new_page()
                shot.set_viewport_size({"width": 520, "height": 300})
                shot.goto(f"http://127.0.0.1:{PORT}/blspl/login?img=cap-4ot0qvfu3q5.jpg")
                shot.wait_for_selector("captchosa-badge", state="attached", timeout=10000)
                shot.wait_for_timeout(800)
                shot.screenshot(path=str(OUT / "page-badge.png"), clip={"x": 0, "y": 70, "width": 520, "height": 170})
                print("wrote page badge")
            finally:
                ctx.close()
    finally:
        fixtures.terminate()


if __name__ == "__main__":
    main()
