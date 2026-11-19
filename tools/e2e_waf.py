"""
End-to-end check of a running aws_capsolver service against an AWS WAF endpoint
you operate (or are explicitly authorised to test).

For each attempt: createTask -> poll getTaskResult -> open the target in Chrome
(Playwright) with the returned aws-waf-token cookie and the same User-Agent, and
count the attempt as a success when the page no longer shows the WAF interstitial.

    python tools/e2e_waf.py --url https://my-waf-test.example/ --key KEY --attempts 5
"""

import argparse
import json
import statistics
import sys
import time
from urllib.parse import urlparse

import requests

WAF_MARKERS = ("gokuProps", "captcha.awswaf.com", "AwsWafIntegration")


def solve(service: str, key: str, url: str, proxy: str = None, timeout: float = 180) -> dict:
    task = {"type": "AntiAwsWafTask" if proxy else "AntiAwsWafTaskProxyLess", "websiteURL": url}
    if proxy:
        task["proxy"] = proxy
    created = requests.post(f"{service}/createTask", json={"clientKey": key, "task": task},
                            timeout=30).json()
    if created.get("errorId"):
        return created
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        time.sleep(1.5)
        res = requests.post(f"{service}/getTaskResult",
                            json={"clientKey": key, "taskId": created["taskId"]}, timeout=30).json()
        if res.get("errorId") or res.get("status") in ("ready", "failed"):
            return res
    return {"errorId": 1, "errorCode": "CLIENT_TIMEOUT"}


def check_in_chrome(url: str, cookie: str, user_agent: str, headless: bool = True) -> bool:
    from playwright.sync_api import sync_playwright
    host = urlparse(url).hostname
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=headless)
        try:
            context = browser.new_context(user_agent=user_agent)
            context.add_cookies([{"name": "aws-waf-token", "value": cookie, "domain": host,
                                  "path": "/", "secure": True}])
            page = context.new_page()
            resp = page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            html = page.content()
            return bool(resp) and resp.status not in (202, 403, 405) and \
                not any(m in html for m in WAF_MARKERS)
        finally:
            browser.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="AWS WAF protected URL you are authorised to test")
    ap.add_argument("--service", default="http://127.0.0.1:8790")
    ap.add_argument("--key", required=True)
    ap.add_argument("--proxy")
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--pause", type=float, default=5.0)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    results = []
    for n in range(1, min(args.attempts, 30) + 1):
        t0 = time.monotonic()
        res = solve(args.service, args.key, args.url, args.proxy)
        elapsed = time.monotonic() - t0
        ok = False
        if res.get("status") == "ready":
            sol = res["solution"]
            ok = bool(sol.get("cookie")) and check_in_chrome(args.url, sol["cookie"],
                                                             sol["userAgent"], not args.headed)
        results.append({"attempt": n, "ok": ok, "seconds": round(elapsed, 1),
                        "error": res.get("errorCode")})
        print(json.dumps(results[-1]), flush=True)
        time.sleep(args.pause)
    good = [r["seconds"] for r in results if r["ok"]]
    print(json.dumps({"attempts": len(results), "success": len(good),
                      "success_rate": round(len(good) / len(results), 3),
                      "avg_seconds_ok": round(statistics.mean(good), 1) if good else None}))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
