"""
End-to-end test of the captchosa extension in Chromium (Playwright, --load-extension).

Only local pages are visited: the fixture server (test/fixture_server.py) and the real
captchosa service started here on 127.0.0.1. Run from the repository root:

    .venv/bin/python extension/test/run_e2e.py [--headed] [--grids 4]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
REPO = EXT.parent
PY = sys.executable
SERVICE_PORT = 8791
FIXTURE_PORT = 8792
SERVICE = f"http://127.0.0.1:{SERVICE_PORT}"
FIX = f"http://127.0.0.1:{FIXTURE_PORT}"
KEY = "e2e-" + os.urandom(6).hex()
BLS_REAL = Path("/home/gg/projects/bls_capsolver/data/real")
BLSPL_IMAGE, BLSPL_TRUTH = "cap-4ot0qvfu3q5.jpg", "56107"

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}", flush=True)


def http_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def wait_http(url, timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            http_json(url, 2)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def fixture_log():
    return http_json(f"{FIX}/_log")


def wait_until(fn, timeout, step=0.25):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(step)
    return None


def bls_grids(n):
    labels = json.loads((BLS_REAL / "labels.json").read_text())
    out = []
    with open(BLS_REAL / "grids.jsonl") as fh:
        for line in fh:
            g = json.loads(line)
            if all(t in labels for t in g["tiles"]):
                truth = [i for i, t in enumerate(g["tiles"]) if labels[t] == g["target"]]
                if truth:
                    out.append((g["grid"], truth))
            if len(out) >= n:
                break
    return out


class Ext:
    """Talks to the extension through its own options page."""

    def __init__(self, context, ext_id):
        self.context = context
        self.base = f"chrome-extension://{ext_id}"
        self.page = context.new_page()
        self.page.goto(f"{self.base}/ui/options.html")

    def eval(self, js, arg=None):
        return self.page.evaluate(js, arg)

    def settings(self, **patch):
        self.eval("""async (patch) => {
            const { settings } = await chrome.storage.local.get('settings');
            const s = CAPTCHOSA.normalize(settings);
            if (patch.solvers) patch.solvers = Object.assign({}, s.solvers, patch.solvers);
            await chrome.storage.local.set({ settings: CAPTCHOSA.normalize(Object.assign(s, patch)) });
        }""", patch)

    def storage(self, key):
        return self.eval("async (k) => (await chrome.storage.local.get(k))[k]", key)

    def badge_text(self, url_part):
        return self.eval("""async (part) => {
            const tabs = await chrome.tabs.query({});
            const tab = tabs.find(t => (t.url || '').includes(part));
            return tab ? chrome.action.getBadgeText({ tabId: tab.id }) : null;
        }""", url_part)


def badge_covers(frame, input_js):
    """Area (px²) of the in-page pill that lies over the input found by ``input_js``."""
    return frame.evaluate("""(inputJs) => {
        const h = document.querySelector('captchosa-badge');
        const inp = eval(inputJs);
        if (!h || !h.shadowRoot || !inp) return -1;
        const a = h.shadowRoot.querySelector('.pill').getBoundingClientRect();
        const b = inp.getBoundingClientRect();
        return Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)) *
               Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
    }""", input_js)


def page_badge(page):
    """(state class, message, solve button visible) of the in-page pill, searched in all frames."""
    for frame in page.frames:
        try:
            got = frame.evaluate("""() => {
                const h = document.querySelector('captchosa-badge');
                if (!h || !h.shadowRoot) return null;
                const p = h.shadowRoot.querySelector('.pill');
                const b = h.shadowRoot.querySelector('button');
                const cls = p.className.split(' ').filter((c) => c !== 'compact' && c !== 'dragging').join(' ');
                return [cls, h.shadowRoot.querySelector('.msg').textContent, !b.hidden];
            }""")
        except Exception:
            got = None
        if got:
            return got
    return None


def run(args):
    tmp = Path(tempfile.mkdtemp(prefix="captchosa-e2e-"))
    (tmp / "keys.json").write_text(json.dumps({KEY: {"name": "e2e"}}))
    env = dict(os.environ, AWSCS_KEYS_FILE=str(tmp / "keys.json"), AWSCS_USAGE_DB=str(tmp / "usage.db"),
               AWSCS_WARMUP="0", PYTHONUNBUFFERED="1")
    service = subprocess.Popen([PY, "-m", "aws_capsolver", "--log-level", "WARNING", "serve",
                                "--port", str(SERVICE_PORT)], cwd=REPO, env=env,
                               stdout=open(tmp / "service.log", "w"), stderr=subprocess.STDOUT)
    fixtures = subprocess.Popen([PY, str(HERE / "fixture_server.py"), "--port", str(FIXTURE_PORT)],
                                stdout=open(tmp / "fixtures.log", "w"), stderr=subprocess.STDOUT)
    try:
        if not wait_http(f"{SERVICE}/health", 60) or not wait_http(f"{FIX}/_log", 15):
            check("servers start", False, f"see {tmp}")
            return
        ext_dir = EXT
        if args.grant_all_urls:
            # a copy whose manifest grants <all_urls> up front (stands in for the user's
            # "Allow on all sites" click, which needs a browser prompt)
            ext_dir = tmp / "ext"
            shutil.copytree(EXT, ext_dir, ignore=shutil.ignore_patterns("dist", "test", "docs"))
            man = json.loads((ext_dir / "manifest.json").read_text())
            man["host_permissions"] = list(man.get("host_permissions", [])) + ["<all_urls>"]
            (ext_dir / "manifest.json").write_text(json.dumps(man, indent=2))
        with sync_playwright() as pw:
            context = pw.chromium.launch_persistent_context(
                str(tmp / "profile"), headless=not args.headed, channel="chromium",
                args=[f"--disable-extensions-except={ext_dir}", f"--load-extension={ext_dir}"],
                viewport={"width": 1100, "height": 900})
            try:
                scenarios(context, args)
            finally:
                context.close()
    finally:
        for p in (service, fixtures):
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
    print(f"\nlogs: {tmp}")


def scenarios(context, args):
    sw = context.service_workers[0] if context.service_workers else context.wait_for_event("serviceworker", timeout=15000)
    ext_id = sw.url.split("/")[2]
    check("extension loaded", bool(ext_id), ext_id)
    time.sleep(1)
    for p in list(context.pages):  # the welcome options tab opened on install
        if p.url.startswith("chrome-extension://"):
            p.close()
    ext = Ext(context, ext_id)
    opt = ext.page

    # ---- options page: configure, save, test connection
    opt.fill("#serviceUrl", SERVICE)
    opt.fill("#clientKey", KEY)
    opt.click("#save")
    opt.wait_for_function("document.querySelector('#message').textContent.length > 0", timeout=10000)
    opt.click("#test")
    ok = wait_until(lambda: "Connected" in opt.inner_text("#message"), 20)
    check("options: save + test connection", ok, opt.inner_text("#message"))
    check("options: balance shown", opt.inner_text("#balance") not in ("", "—"), opt.inner_text("#balance"))
    s = ext.storage("settings")
    check("settings stored", s["serviceUrl"] == SERVICE and s["clientKey"] == KEY)
    # the delay slider: keyboard End then Home+steps would be slow; set it like a user drag would
    opt.evaluate("""() => { const r = document.getElementById('delayMs'); r.value = 200;
        r.dispatchEvent(new Event('input')); r.dispatchEvent(new Event('change')); }""")
    time.sleep(0.3)
    check("delay slider saves", ext.storage("settings")["delayMs"] == 200, opt.inner_text("#delayOut"))
    opt.click("#retriesUp")
    time.sleep(0.3)
    up = ext.storage("settings")["retries"]
    opt.click("#retriesDown")
    time.sleep(0.3)
    check("retries stepper", up == 4 and ext.storage("settings")["retries"] == 3, str(up))

    if args.only == "geetest":
        geetest_scenarios(context, ext)
        ext.settings(serviceUrl=SERVICE)
        geetest_real_service(context)
        return

    # ---- AWS WAF grid: open shadow root, fetch, first answer rejected (2 rounds)
    page = context.new_page()
    page.goto(f"{FIX}/fixtures/aws_shadow.html")
    solved = wait_until(lambda: page.title() == "AWS SOLVED", 120)
    verifies = fixture_log()["verify"]
    check("aws shadow: solved after a rejected round", solved and len(verifies) == 2 and
          all(v["solution"] == [0, 2, 4, 6, 8] for v in verifies) and verifies[-1]["accepted"],
          json.dumps(verifies))
    check("toolbar badge shows success", ext.badge_text("aws_shadow") in ("✓", ""), repr(ext.badge_text("aws_shadow")))
    page.close()

    # ---- AWS WAF grid in a cross-origin iframe, closed shadow root, XHR transport
    page = context.new_page()
    page.goto(f"{FIX}/fixtures/aws_iframe.html")
    got = wait_until(lambda: len(fixture_log()["verify"]) >= 3, 120)
    v = fixture_log()["verify"][2:] if got else []
    check("aws iframe+closed shadow+xhr: correct tiles", got and v[0]["solution"] == [0, 2, 4, 6, 8] and v[0]["accepted"],
          json.dumps(v))
    page.close()

    # ---- AWS manual mode: no auto solve, Solve button next to the captcha (live setting)
    ext.settings(autoSolve=False)
    page = context.new_page()
    page.goto(f"{FIX}/fixtures/aws_frame.html")
    badge = wait_until(lambda: page_badge(page), 20)
    time.sleep(2)
    n_before = len(fixture_log()["verify"])
    check("manual mode: Solve button shown, nothing sent", badge and badge[2] and n_before == 3, repr(badge))
    page.locator("captchosa-badge").locator("button.solve").click()
    got = wait_until(lambda: len(fixture_log()["verify"]) > n_before, 90)
    v = (fixture_log()["verify"] or [{}])[-1]
    check("manual mode: Solve button solves", got and v["solution"] == [0, 2, 4, 6, 8] and v["accepted"], json.dumps(v))
    page.close()
    ext.settings(autoSolve=True)

    # ---- AWS WAF token: gokuProps page -> AntiAwsWafTaskProxyLess (service refuses a private URL)
    page = context.new_page()
    page.goto(f"{FIX}/fixtures/aws_token.html")
    badge = wait_until(lambda: (page_badge(page) or [""])[0].endswith("error") and page_badge(page), 60)
    stats = ext.storage("stats") or {}
    check("aws token: task sent, typed error shown", badge and ":" in badge[1] and stats.get("awsToken", {}).get("failed", 0) >= 1,
          repr(badge))
    check("toolbar badge shows error", ext.badge_text("aws_token") == "ERR", repr(ext.badge_text("aws_token")))
    page.close()

    # ---- BLS grid (real tiles, decoys hidden 5 different ways), auto submit
    grids = bls_grids(args.grids)
    for grid_id, truth in grids:
        page = context.new_page()
        n = len(fixture_log()["bls_submit"])
        page.goto(f"{FIX}/bls/grid?grid={grid_id}&variant=appointment")
        got = wait_until(lambda: len(fixture_log()["bls_submit"]) > n, 90)
        sub = fixture_log()["bls_submit"][-1] if got else None
        check(f"bls grid {grid_id}: clicked = truth {truth}", sub and sub["selected"] == truth,
              json.dumps(sub))
        page.close()

    # ---- BLS login variant: empty password -> pick tiles but do not submit
    grid_id, truth = grids[0]
    page = context.new_page()
    n = len(fixture_log()["bls_submit"])
    page.goto(f"{FIX}/bls/grid?grid={grid_id}&variant=login")
    badge = wait_until(lambda: (page_badge(page) or [""])[0].endswith("ok") and page_badge(page), 90)
    picked = page.evaluate("[...document.querySelectorAll('img.img-selected')].map(i => i.dataset.fx)")
    time.sleep(1.5)
    check("bls login: tiles picked, no submit with empty password",
          sorted(int(p[4:]) for p in picked if p.startswith("real")) == truth and len(picked) == len(truth)
          and len(fixture_log()["bls_submit"]) == n, f"{picked} {badge}")
    page.close()

    # ---- global switch off: nothing happens; toolbar shows OFF
    ext.settings(enabled=False)
    time.sleep(0.5)
    page = context.new_page()
    page.goto(f"{FIX}/bls/grid?grid={grids[1][0]}&variant=login")
    time.sleep(4)
    picked = page.evaluate("document.querySelectorAll('img.img-selected').length")
    global_badge = ext.eval("chrome.action.getBadgeText({})")
    check("disabled: no clicks, toolbar OFF", picked == 0 and global_badge == "OFF", f"{picked} {global_badge!r}")
    # ---- live: switching on again solves the open page without a reload
    ext.settings(enabled=True)
    got = wait_until(lambda: (page_badge(page) or [""])[0].endswith("ok"), 60)
    picked = page.evaluate("[...document.querySelectorAll('img.img-selected')].map(i => i.dataset.fx)")
    check("live enable: solves without reload", got and sorted(int(p[4:]) for p in picked if p.startswith("real")) == grids[1][1],
          repr(picked))
    page.close()

    # ---- BLS Poland text captcha against a mocked ImageToTextTask
    ext.settings(serviceUrl=f"{FIX}/mockapi")
    page = context.new_page()
    page.goto(f"{FIX}/blspl/login?img={BLSPL_IMAGE}")
    got = wait_until(lambda: page.input_value("#captcha_code") == "40719", 60)
    events = page.evaluate("window.__events")
    log = fixture_log()
    mock = log["mockapi"][-1] if log["mockapi"] else {}
    check("bls text (mock): filled + input/change events", got and "input:40719" in events and "change:40719" in events,
          repr(events))
    check("bls text: image downloaded once (read from the page)", log["blspl_images"].count(BLSPL_IMAGE) == 1,
          repr(log["blspl_images"]))
    check("bls text: task shape", mock.get("type") == "ImageToTextTask" and mock.get("module") == "bls_poland"
          and mock.get("body_len", 0) > 100, json.dumps(mock))
    time.sleep(1)
    check("bls text: no submit by default", not fixture_log()["blspl_submit"])
    page.close()

    ext.settings(submitAfterFill=True)
    page = context.new_page()
    page.goto(f"{FIX}/blspl/login?img=cap-5to9xh6ga98.jpg")
    got = wait_until(lambda: fixture_log()["blspl_submit"], 60)
    check("bls text: submit after fill when enabled", got and got[-1] == "40719", repr(got))
    page.close()
    ext.settings(submitAfterFill=False)

    # ---- General text solver: on by default (mocked service), many page layouts
    page = context.new_page()
    page.goto(f"{FIX}/generic/login")
    got = wait_until(lambda: page.input_value("#verification_code") == "aB3dE", 60)
    mock = fixture_log()["mockapi"][-1]
    check("general text: on by default, solved (module common)", got and mock.get("module") == "common",
          json.dumps(mock))
    check("general text: case-sensitive task (case=true), mixed case filled exactly",
          got and mock.get("case") is True, f"{page.input_value('#verification_code')!r} {json.dumps(mock)}")
    page.close()

    def generic_variant(variant, selector, expect_fill=True):
        page = context.new_page()
        n_mock = len(fixture_log()["mockapi"])
        page.goto(f"{FIX}/generic/v?variant={variant}")
        read = (lambda: page.evaluate("document.getElementById('host').shadowRoot.getElementById('kod').value")) \
            if variant == "shadow" else (lambda: page.input_value(selector))
        if expect_fill:
            got = wait_until(lambda: read() == "aB3dE", 60)
            check(f"general text [{variant}]: detected and filled", got, f"value={read()!r} badge={page_badge(page)}")
            if selector:
                cov = badge_covers(page.main_frame, f"document.querySelector('{selector}')")
                check(f"general text [{variant}]: pill does not cover the input", cov == 0, f"overlap={cov}")
            if variant == "label":
                pill_box = lambda: page.evaluate("""() => { const r = document.querySelector('captchosa-badge')
                    .shadowRoot.querySelector('.pill').getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; }""")
                x, y, w, h = pill_box()
                page.mouse.move(x + 8, y + h / 2)
                page.mouse.down()
                page.mouse.move(x + 8 - 120, y + h / 2 + 40, steps=8)
                page.mouse.up()
                time.sleep(1.2)  # survives the periodic reposition
                x2, y2, _, _ = pill_box()
                check("badge: can be dragged (and stays there)", abs((x2 - x) + 120) <= 3 and abs((y2 - y) - 40) <= 3,
                      f"{(x, y)} -> {(x2, y2)}")
                page.evaluate("document.querySelector('captchosa-badge').shadowRoot.querySelector('button.close').click()")
                time.sleep(1.2)
                shown = page.evaluate("getComputedStyle(document.querySelector('captchosa-badge')).display")
                check("badge: x hides it", shown == "none", shown)
        else:
            time.sleep(5)
            check(f"general text [{variant}]: nothing detected or filled",
                  read() == "" and len(fixture_log()["mockapi"]) == n_mock and page_badge(page) is None,
                  f"value={read()!r} badge={page_badge(page)}")
        page.close()

    for variant, selector in (("label", "#ans"), ("canvas", "#sec"), ("shadow", None)):
        generic_variant(variant, selector)
    generic_variant("negative", "#user", expect_fill=False)

    # cross-origin picture: needs the <all_urls> grant (a browser prompt) for the screenshot path,
    # so without it the pill must show a clear CAPTURE_DENIED; with it the field is filled
    page = context.new_page()
    page.goto(f"{FIX}/generic/v?variant=cross")
    if args.grant_all_urls:
        got = wait_until(lambda: page.input_value("#xin") == "aB3dE", 60)
        check("general text [cross]: read from a screenshot and filled", got,
              f"value={page.input_value('#xin')!r} badge={page_badge(page)}")
    else:
        got = wait_until(lambda: "CAPTURE" in ((page_badge(page) or ["", ""])[1]), 60)
        check("general text [cross]: detected, clear CAPTURE_DENIED without the grant", got,
              f"value={page.input_value('#xin')!r} badge={page_badge(page)}")
    page.close()

    # NopeCHA-demo layout: 12 iframes (src set by script), .captchapict + .captcha only
    page = context.new_page()
    page.goto(f"{FIX}/generic/nopecha?n=12")

    def nopecha_filled():
        vals = []
        for fr in page.frames[1:]:
            try:
                vals.append(fr.evaluate("(document.querySelector('input.captcha') || {}).value || ''"))
            except Exception:
                vals.append("")
        return vals
    got = wait_until(lambda: (lambda v: len(v) == 12 and all(x == "aB3dE" for x in v))(nopecha_filled()), 120)
    check("general text [nopecha-like 12 iframes]: every captcha filled", got, repr(nopecha_filled()))
    page.close()

    # the real demo writes its iframes with document.write (about:blank frames)
    page = context.new_page()
    page.goto(f"{FIX}/generic/nopecha_docwrite?n=8")
    got = wait_until(lambda: (lambda v: len(v) == 8 and all(x == "aB3dE" for x in v))(nopecha_filled()), 120)
    check("general text [nopecha document.write iframes]: every captcha filled", got, repr(nopecha_filled()))
    covers = [badge_covers(fr, "document.querySelector('input.captcha')") for fr in page.frames[1:]]
    check("general text [nopecha document.write iframes]: pill never covers the input",
          all(c == 0 for c in covers), repr(covers))
    page.close()

    geetest_scenarios(context, ext)

    # switching the general solver off stops it live
    ext.settings(solvers={"text": False})
    page = context.new_page()
    n_mock = len(fixture_log()["mockapi"])
    page.goto(f"{FIX}/generic/login")
    time.sleep(4)
    check("general text: off when switched off", page.input_value("#verification_code") == ""
          and len(fixture_log()["mockapi"]) == n_mock)
    page.close()
    ext.settings(solvers={"text": True}, serviceUrl=SERVICE)

    # ---- BLS Poland text against the real service (model may be absent: then a typed error)
    page = context.new_page()
    page.goto(f"{FIX}/blspl/login?img={BLSPL_IMAGE}")
    got = wait_until(lambda: page.input_value("#captcha_code") or
                     ((page_badge(page) or [""])[0].endswith("error") and page_badge(page)), 90)
    value = page.input_value("#captcha_code")
    check("bls text (real service): answered", bool(got),
          f"filled {value!r} (truth {BLSPL_TRUTH})" if value else f"badge {page_badge(page)!r}")
    page.close()

    geetest_real_service(context)

    # ---- popup reflects counters and log
    opt.reload()
    opt.wait_for_selector("#log li")
    chips = opt.locator("#solvers .chip.ok").all_inner_texts()
    n_log = opt.locator("#log li").count()
    check("options: counters + log rendered", any(c != "✓ 0" for c in chips) and n_log >= 5, f"{chips} {n_log} log rows")
    popup = context.new_page()
    popup.set_viewport_size({"width": 360, "height": 700})
    popup.goto(f"{ext.base}/ui/popup.html")
    popup.wait_for_selector("#statusPill.ok", timeout=15000)
    check("popup: connected pill + balance", popup.inner_text("#balance") not in ("", "—"),
          popup.inner_text("#statusPill"))


def geetest_scenarios(context, ext):
    """GeeTest v3 / v4 fixture pages against the mocked GeeTestClassification (service = mock)."""

    ext.settings(serviceUrl=f"{FIX}/mockapi")

    def gt_logs(page_name, type_, since):
        return [e for e in fixture_log()["geetest"][since:] if e.get("page") == page_name and e.get("type") == type_]

    def run_page(version, type_, mode=""):
        since = len(fixture_log()["geetest"])
        n_mock = len(fixture_log()["mockapi"])
        imgs_before = list(fixture_log()["geetest_images"])
        page = context.new_page()
        page.goto(f"{FIX}/fixtures/geetest_{version}.html?type={type_}&mode={mode}")
        ok = wait_until(lambda: page.title() == "GEETEST OK", 90)
        if ok:  # the pill turns green once the solver has seen the verdict
            wait_until(lambda: (page_badge(page) or [""])[0].endswith("ok"), 5)
        logs = [e for e in gt_logs(version, type_, since) if e.get("event") != "open"]
        mocks = [m for m in fixture_log()["mockapi"][n_mock:] if m.get("type") == "GeeTestClassification"]
        imgs = fixture_log()["geetest_images"][len(imgs_before):]
        badge = page_badge(page)
        page.close()
        return ok, logs, mocks, imgs, badge

    # v3 slide: canvases, 1:1 slider, the first answer is rejected -> second round accepted
    ok, logs, mocks, _, badge = run_page("v3", "slide", "wrongfirst")
    last = logs[-1] if logs else {}
    check("geetest v3 slide: solved in 2 rounds (first rejected)", ok and len(logs) == 2 and last.get("accepted"),
          json.dumps(logs))
    check("geetest v3 slide: piece over the gap at x=172 (+-3 px)", ok and abs(last.get("final", 0) - 172) <= 3,
          json.dumps(last))
    check("geetest v3 slide: human-like drag (many moves, >300 ms, vertical wobble, overshoot)",
          last.get("moves", 0) >= 15 and last.get("ms", 0) >= 300 and last.get("yRange", 0) > 0 and last.get("backtrack"),
          json.dumps(last))
    m = mocks[0] if mocks else {}
    check("geetest v3 slide: task shape (bg + piece cut from the slice canvas + full background)",
          m.get("question") == "geetest:slide" and m.get("size") == [260, 160] and (m.get("piece") or [0])[0] in (54, 55) and (m.get("piece") or [0, 0])[1] == 160
          and m.get("fullBackground") is True, json.dumps(mocks))
    check("geetest v3 slide: pill shows success", badge and badge[0].endswith("ok"), repr(badge))

    # v3 word click: one <img> with a hint strip, scaled to 306 px; clicks in order, then commit
    ok, logs, mocks, imgs, _ = run_page("v3", "click")
    last = logs[-1] if logs else {}
    check("geetest v3 click: 3 points clicked in order + commit", ok and last.get("accepted"), json.dumps(logs))
    m = mocks[0] if mocks else {}
    check("geetest v3 click: word task with the hint characters",
          m.get("question") == "geetest:word" and m.get("hints") == ["天", "地", "人"] and m.get("size") == [344, 384],
          json.dumps(m))
    check("geetest v3 click: picture read from the page (downloaded once)", imgs.count("v3click") == 1, repr(imgs))

    # v4 slide: CSS background pictures, hashed classes, pointer events, piece/handle ratio != 1
    ok, logs, mocks, imgs, _ = run_page("v4", "slide")
    last = logs[-1] if logs else {}
    check("geetest v4 slide: piece over the gap at x=172 (+-3 px), first round", ok and len(logs) == 1 and
          abs(last.get("final", 0) - 172) <= 3, json.dumps(logs))
    m = mocks[0] if mocks else {}
    check("geetest v4 slide: task shape (bg 320x160 + piece 64x64)",
          m.get("size") == [320, 160] and m.get("piece") == [64, 64], json.dumps(m))
    check("geetest v4 slide: pictures not downloaded again", imgs.count("v4bg") == 1 and imgs.count("v4piece") == 1,
          repr(imgs))

    # v4 icon: hint icons sent as pictures, clicks on a 600x400 picture shown at 300x200
    ok, logs, mocks, _, _ = run_page("v4", "icon")
    check("geetest v4 icon: clicks in order + submit", ok and logs and logs[-1].get("accepted"), json.dumps(logs))
    m = mocks[0] if mocks else {}
    check("geetest v4 icon: task with 3 hint pictures", m.get("question") == "geetest:icon" and
          m.get("hints") == [[30, 30]] * 3 and m.get("size") == [600, 400], json.dumps(m))

    # v4 nine-grid: 9 cell pictures drawn into one grid picture, target picture
    ok, logs, mocks, _, _ = run_page("v4", "nine")
    check("geetest v4 nine: picked cells 1,5,6 + submit", ok and logs and logs[-1].get("picked") == [1, 5, 6],
          json.dumps(logs))
    m = mocks[0] if mocks else {}
    check("geetest v4 nine: grid picture 300x300 + target picture", m.get("question") == "geetest:nine" and
          m.get("size") == [300, 300] and m.get("targetImage") == [30, 30], json.dumps(m))

    # v4 gobang (winlinze): 5x5 board, move (0,1) -> (2,3)
    ok, logs, mocks, _, _ = run_page("v4", "gobang")
    check("geetest v4 gobang: move clicked", ok and logs and logs[-1].get("clicks") == [[0, 1], [2, 3]], json.dumps(logs))
    m = mocks[0] if mocks else {}
    check("geetest v4 gobang: board picture 300x300", m.get("question") == "geetest:gobang" and m.get("size") == [300, 300],
          json.dumps(m))

    # the GeeTest switch: off -> the widget is not even opened
    ext.settings(solvers={"geetest": False})
    since = len(fixture_log()["geetest"])
    n_mock = len(fixture_log()["mockapi"])
    page = context.new_page()
    page.goto(f"{FIX}/fixtures/geetest_v4.html?type=icon")
    time.sleep(4)
    check("geetest: off when switched off", not fixture_log()["geetest"][since:] and
          len(fixture_log()["mockapi"]) == n_mock and page.title() != "GEETEST OK")
    page.close()
    ext.settings(solvers={"geetest": True})


def geetest_real_service(context):
    """Against the real service the model may be missing: a typed error in the pill, or solved."""
    since = len(fixture_log()["geetest"])
    page = context.new_page()
    page.goto(f"{FIX}/fixtures/geetest_v4.html?type=slide")
    got = wait_until(lambda: page.title() == "GEETEST OK" or
                     ((page_badge(page) or [""])[0].endswith("error") and page_badge(page)), 90)
    badge = page_badge(page)
    drops = [e for e in fixture_log()["geetest"][since:] if e.get("event") == "verify"]
    # with the model installed the answer is applied (the synthetic picture may still fool it);
    # without it the pill shows the service's typed error
    check("geetest (real service): answer applied, or a typed error", bool(got) and (
          page.title() == "GEETEST OK" or bool(drops) or (badge and badge[1].split(":")[0].startswith("ERROR_"))),
          f"{badge!r} drops={json.dumps(drops)}")
    page.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--grids", type=int, default=4)
    ap.add_argument("--grant-all-urls", action="store_true",
                    help="load a copy of the extension with <all_urls> granted (cross-origin captcha path)")
    ap.add_argument("--only", choices=("geetest",), help="run only one group of scenarios")
    args = ap.parse_args()
    run(args)
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed or not results else 0)


if __name__ == "__main__":
    main()
