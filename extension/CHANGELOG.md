# Changelog

All notable changes to the Captchosa browser extension. Versions follow
[semantic versioning](https://semver.org/); bump with `python3 tools/build_extension.py patch|minor|major`.

## 1.0.0 – 2026-11-20

First release.

### Solvers
- **AWS WAF grid** – reads the widget's own `/problem` answer (fetch + XHR observed in the page's
  main world, also in iframes and closed shadow roots), sends `AwsWafClassification`, clicks the
  tiles and Confirm like a person; follows wrong-answer rounds up to the retry limit.
- **AWS WAF token** – on challenge pages with `gokuProps`, sends `AntiAwsWafTaskProxyLess` with the
  browser's user agent and sets the `aws-waf-token` cookie (reload-loop guard).
- **BLS grid** – finds the visible target label and the 9 visible tiles (decoys hidden by CSS,
  off-screen or covered are skipped by hit-testing), sends `BlsClassification`, clicks the tiles and
  presses Verify unless a password field is still empty.
- **BLS Poland text** – reads the already loaded captcha image through a canvas (never downloads it
  again), sends `ImageToTextTask` (module `bls_poland`) and fills `captcha_code`.
- **General text** (off by default) – any image captcha next to a captcha/code input, module `common`.

### Interface
- Popup: global switch, per-solver switches with solved/failed counters and last solve,
  connection + balance pill, auto-solve, delay slider, retries, submit-after-fill, recent activity.
- Options page with the same design: service URL, client key (runtime host permission for
  non-localhost URLs), test connection, advanced settings, counter reset.
- On-page badge with a Solve button in manual mode; toolbar icon + badge show on/off/solving/error.
- Light/dark themes from the system; English, Turkish, Azerbaijani and Russian (chrome.i18n).

### Tooling
- `tools/build_extension.py` – validation (files, locales, forbidden patterns), version bump, zip.
- `tools/make_icons.py`, `tools/locales.py` – icons from the SVG mark, locale files from one table.
- `test/run_e2e.py` – Playwright end-to-end test against local fixtures and a local service.
