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


def _post(service: str, path: str, body: dict) -> dict:
    """POST to the service; transport failures become an error dict instead of a crash."""
    try:
        resp = requests.post(f"{service}{path}", json=body, timeout=30)
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        return {"errorId": 1, "errorCode": "CLIENT_ERROR",
                "errorDescription": exc.__class__.__name__}
    return data if isinstance(data, dict) else {"errorId": 1, "errorCode": "CLIENT_ERROR"}


def solve(service: str, key: str, url: str, proxy: str = None, timeout: float = 180) -> dict:
    task = {"type": "AntiAwsWafTask" if proxy else "AntiAwsWafTaskProxyLess", "websiteURL": url}
    if proxy:
        task["proxy"] = proxy
    created = _post(service, "/createTask", {"clientKey": key, "task": task})
    if created.get("errorId") or not created.get("taskId"):
        return created
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        time.sleep(1.5)
        res = _post(service, "/getTaskResult", {"clientKey": key, "taskId": created["taskId"]})
        if res.get("errorCode") == "CLIENT_ERROR":
            continue  # transient: keep polling until the timeout
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
    for n in range(1, max(1, min(args.attempts, 30)) + 1):
        t0 = time.monotonic()
        res = solve(args.service, args.key, args.url, args.proxy)
        elapsed = time.monotonic() - t0
        ok = False
        sol = res.get("solution") or {}
        if res.get("status") == "ready" and sol.get("cookie"):
            try:
                ok = check_in_chrome(args.url, sol["cookie"], sol.get("userAgent") or "",
                                     not args.headed)
            except Exception as exc:  # browser problems must not abort the whole run
                print(json.dumps({"attempt": n, "browser_error": exc.__class__.__name__}))
        results.append({"attempt": n, "ok": ok, "seconds": round(elapsed, 1),
                        "error": res.get("errorCode")})
        print(json.dumps(results[-1]), flush=True)
        time.sleep(max(0.0, args.pause))
    good = [r["seconds"] for r in results if r["ok"]]
    print(json.dumps({"attempts": len(results), "success": len(good),
                      "success_rate": round(len(good) / len(results), 3),
                      "avg_seconds_ok": round(statistics.mean(good), 1) if good else None}))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
