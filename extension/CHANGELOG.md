# Changelog

All notable changes to the Captchosa browser extension. Versions follow
[semantic versioning](https://semver.org/); bump with `python3 tools/build_extension.py patch|minor|major`.

## 1.0.2 – 2026-11-21

### Fixed
- **Text captchas were not detected on most sites.** The general text solver was off by default
  (stored settings are migrated once to *on*), and both text solvers checked the BLS Poland switch
  instead of their own.
- Detection is much broader: `<canvas>` captchas, `input[type=image]`, captcha background images,
  open shadow roots, inputs recognised by their `<label>` / surrounding text (en/tr/az/ru), and
  pictures recognised by shape when they sit next to a captcha input. Ordinary login, search and
  e-mail fields are excluded.
- Cross-origin captcha pictures are read from a screenshot of the visible tab (new
  **Allow on all sites** button in Options → Advanced).

### Changed
- The input shows *Solving…* (localised) while the answer is on its way.
- Text tasks are sent with `case: true`: the answer keeps upper/lower case exactly as read.

### Tests
- New end-to-end pages: label-only, canvas, shadow DOM + Turkish wording, a negative login page,
  a cross-origin picture (`run_e2e.py --grant-all-urls` covers the screenshot path), a
  NopeCHA-demo-like page (12 script-filled iframes with `.captchapict` + `.captcha` only), and a
  mixed-case answer that must reach the input unchanged.

## 1.0.1 – 2026-11-21

### Fixed
- **AWS WAF grid did not solve on the real widget.** The real widget lays its 9
  `button.amzn-grid-cell` cells over a `<canvas>` in the same parent; the cell finder grouped
  siblings by the first child's tag (the canvas), fell back to clicking the canvas and the
  widget submitted an empty answer. Cells are now found by `.amzn-grid-cell` first and
  siblings are grouped per tag.
- **No automatic start.** The widget fetches no puzzle until **Begin** is pressed; with
  auto-solve on, the extension now presses it (after the configured delay).
- On CAPTCHA pages the token flow no longer runs first and blocks the in-page grid solve; it is
  kept for challenge-only pages.
- The confirm step prefers the widget's own `#amzn-captcha-verify-button`.

### Tests
- `test/real_widget.py`: AWS's own `captcha.js` (local copy, not committed) served on
  127.0.0.1 with a mocked `/problem` + `/verify`; the extension must press Begin, pick the
  tiles and submit unaided.

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
