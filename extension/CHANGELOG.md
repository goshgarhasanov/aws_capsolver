# Changelog

All notable changes to the Captchosa browser extension. Versions follow
[semantic versioning](https://semver.org/); bump with `python3 tools/build_extension.py patch|minor|major`.

## 1.0.5 – 2026-11-24

### Added
- **reCAPTCHA v2 image challenge** solver (`content/recaptcha.js`, new *reCAPTCHA v2* switch with
  its own icon, on by default; settings v4) through the service's new synchronous
  `ReCaptchaV2Classification` task (3 tokens per picture).
  - Runs only in reCAPTCHA frames (`…/recaptcha/api2|enterprise/bframe` and `/anchor` on
    www.google.com or www.recaptcha.net); every other frame returns at once.
  - Reads the object from the instruction (`strong`, any language) and the payload picture
    (`img.rc-image-tile-33` / `-44`) from the page, clicks the returned cells with paced pointer
    clicks, classifies every replaced tile of a dynamic 3x3 (`img.rc-image-tile-11`, `size: 1`)
    until none shows the object, then presses Verify.
  - "Please try again" / a new picture = a wrong round; an unsupported object or no answer =
    reload; at most 5 failed rounds and 12 pictures per challenge, then the pill offers *Solve*.
  - With *Auto-solve* the "I'm not a robot" checkbox is clicked once (again after it expired).
    When it turns ticked, the checkbox frame tells the challenge frame through the service worker
    (`recaptchaChecked` relay to the tab's frames): that is the success signal, because Chrome
    stops updating a hidden cross-origin challenge frame.
  - No tokens are produced or injected, nothing changes the browser fingerprint: the visible
    challenge is solved like a user would.

### Permissions
- **No new permissions.** The content scripts already run in every http(s) frame (`all_frames`,
  `match_origin_as_fallback`), which includes Google's reCAPTCHA iframes, and inside the challenge
  frame the payload picture is same-origin, so it is read from the loaded `<img>` (or, if that
  fails, fetched from the frame's own origin) without host access to google.com.

### Tests
- New reCAPTCHA fixture (`test/fixtures/recaptcha.html` + cross-origin `recaptcha_anchor.html` /
  `recaptcha_bframe.html`) with a mocked `ReCaptchaV2Classification`: static 3x3, "Please try
  again", unsupported object (reload), 4x4, dynamic 3x3, the switch, and the real service.

## 1.0.4 – 2026-11-21

### Fixed (verified on the live GeeTest demo at 2captcha.com, v3 4/4 and v4 6/6 solved)
- **GeeTest v3 never solved on real pages.** It answered while the popup still showed
  "Loading" (empty 300x150 canvases -> `piece does not fit into background`); it now waits until
  the background canvas is painted and the slice canvas holds the piece (no round is spent).
- **v3 drag ran away.** The real v3 slice canvas covers the whole picture and the piece is
  redrawn inside it (the element never moves), so the closed loop saw "no movement" and kept
  dragging; the piece position is now read from the canvas content.
- **False "solved".** Real v3 keeps `.geetest_success_radar_tip` & co. in the DOM all the time;
  success now comes only from the state classes.
- **A rejected v3 answer was reported as solved.** When GeeTest answers `forbidden` it resets
  the widget to "Click to verify"; a closed popup without a success state now counts as
  rejected, not solved.
- **v4 landed ~4 px off.** The live v4 widget moves the piece about 4 px when the button is
  released and verifies that position; the drag now aims short by that jump (re-measured on
  every drag of the page).

### Changed
- **One "Text captcha" solver.** BLS Poland text and the general text solver are merged into a
  single switch, counter and icon (the BLS Poland login still uses its own model,
  `module: bls_poland`; every other picture uses `common`). Settings v3 migration: on when either
  old switch was on.

## 1.0.3 – 2026-11-21

### Added
- **GeeTest v3 and v4** solver (`content/geetest.js`, new *GeeTest* switch, on by default):
  slide, icon / word / space click, nine-grid and gobang (winlinze), through the service's new
  synchronous `GeeTestClassification` task (8 tokens).
  - Detects the v3 popup (`.geetest_canvas_bg`, `.geetest_slider_button`, `.geetest_item_img`,
    `.geetest_commit`) and the v4 box (`.geetest_bg`, `.geetest_btn`, `.geetest_item`,
    `.geetest_submit`, also with hashed class suffixes); presses the "click to verify" button
    once with *Auto-solve*.
  - Reads the pictures from the page (canvas pixels, loaded `<img>`, CSS background pictures from
    the memory cache, CORS for GeeTest's CDN) and falls back to a tab screenshot only when needed.
  - Slider: closed-loop, minimum-jerk drag with vertical wobble, overshoot and settle-back,
    pointer + mouse events. Clicks: paced, in the returned order, then Confirm / Submit.
  - Slide tasks carry `y` (the piece image's top = GeeTest's `ypos`, from the v3 slice canvas or
    the v4 piece position), which lets the model rule out v4 decoy gaps.
  - A fresh puzzle after the answer is a wrong round (up to *Retries*); success or the widget
    closing counts as solved. The pill, the toolbar badge and the activity log show each step.

### Tests
- New GeeTest fixture pages (`test/fixtures/geetest_v3.html`, `geetest_v4.html`) with a mocked
  `GeeTestClassification`: v3 slide (first round rejected), v3 word click, v4 slide (piece/handle
  ratio ≠ 1), v4 icon, v4 nine-grid, v4 gobang, the solver switch, and the real service (typed
  error while the model is not installed).

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
- The in-page pill never covers the captcha input (or the focused field): it tries the free
  sides of the picture and of the input, shrinks to its icon in cramped frames, and picks the
  least-overlapping spot as a last resort.
- The pill can be dragged anywhere (it then keeps that offset from the captcha) and has an ×
  that hides it for that captcha; solving goes on.
- Each captcha type in the popup / options list has its own icon (AWS grid, AWS token, BLS
  grid, BLS Poland digits, general text).
- The balance is shown as **tokens** (the service now bills tokens per solve).

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
