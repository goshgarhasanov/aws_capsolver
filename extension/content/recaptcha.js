// reCAPTCHA v2 image challenge ("Select all images with ...": 3x3, 4x4 and the dynamic 3x3
// whose clicked tiles are replaced by new 1x1 pictures).
// Isolated world, only in reCAPTCHA frames: the challenge lives in the cross-origin iframe
// .../recaptcha/(api2|enterprise)/bframe (www.google.com or www.recaptcha.net), the checkbox in
// .../anchor. Inside the bframe the payload picture is same-origin, so it is read from the page
// without any extra permission. The answer is applied like a person would: paced pointer clicks
// on the tiles, then Verify. Nothing here touches tokens, fingerprints or Google's bot checks.
(function () {
  "use strict";
  const CS = globalThis.CS;
  if (!CS || globalThis.__captchosaRecaptcha) return;
  if (window.top === window) return; // the checkbox and the challenge are always iframes
  const path = location.pathname;
  // the real frames, plus local test pages named after them (test/fixtures/recaptcha_*.html)
  const IS_BFRAME = /\/recaptcha\/(api2|enterprise)\/bframe\b/.test(path) || /\/recaptcha_bframe[^/]*$/.test(path);
  const IS_ANCHOR = /\/recaptcha\/(api2|enterprise)\/anchor\b/.test(path) || /\/recaptcha_anchor[^/]*$/.test(path);
  if (!IS_BFRAME && !IS_ANCHOR) return;
  globalThis.__captchosaRecaptcha = true;

  const KIND = "recaptcha";
  const MAX_ROUNDS = 5;          // wrong answers, reloads and "no answer" per session
  const MAX_CHALLENGES = 12;     // pictures per session ("Next" pages included)
  const VERDICT_MS = 7000;       // how long to wait for reCAPTCHA's reaction to Verify
  const IMAGE_TIMEOUT_MS = 10000;
  const DYNAMIC_MS = 90000;      // a dynamic challenge as a whole
  const MAX_DYNAMIC_TILES = 40;
  const ERRORS = {               // reCAPTCHA's own messages (shown with display: block)
    "rc-imageselect-incorrect-response": "try-again",          // "Please try again."
    "rc-imageselect-error-select-more": "select-more",         // "Please select all matching images."
    "rc-imageselect-error-dynamic-more": "dynamic-more",       // "Please also check the new images."
    "rc-imageselect-error-select-something": "select-something",
  };

  const rand = (a, b) => a + Math.random() * (b - a);
  const err = (code, message) => Object.assign(new Error(message), { code });

  const state = {
    busy: false,
    gaveUp: false,
    badge: null,
    frameVisible: typeof IntersectionObserver !== "function", // without IO: assume visible
    anchorClicks: 0,
    anchorChecked: false,
    checkedAt: 0,            // when the checkbox frame last reported "ticked" (via the background)
    done: new Set(),         // signatures of pictures that ended in a solve (the frame keeps them)
  };

  // The parent page hides a closed challenge by moving its iframe off-screen; inside the frame
  // only an IntersectionObserver (against the top-level viewport) can tell.
  function watchVisibility(onChange) {
    if (typeof IntersectionObserver !== "function") return;
    const io = new IntersectionObserver((entries) => {
      const visible = entries.some((e) => e.isIntersecting);
      if (visible !== state.frameVisible) {
        state.frameVisible = visible;
        onChange(visible);
      }
    });
    io.observe(document.body || document.documentElement);
  }

  // ------------------------------------------------------------ challenge frame (bframe)

  function visibleError() {
    for (const [cls, name] of Object.entries(ERRORS)) {
      const el = document.querySelector(`.${cls}`);
      if (el && CS.isVisible(el)) return name;
    }
    return null;
  }

  function questionOf(root) {
    const desc = root.querySelector(".rc-imageselect-desc-no-canonical, .rc-imageselect-desc, .rc-imageselect-desc-wrapper");
    if (!desc) return "";
    const strong = desc.querySelector("strong");
    const text = (strong ? strong.textContent : (desc.firstChild && desc.firstChild.textContent) || desc.textContent) || "";
    return text.replace(/\s+/g, " ").trim().slice(0, 128);
  }

  function cellsOf(table) {
    return [...table.querySelectorAll("td")].filter((td) => td.classList.contains("rc-imageselect-tile") ||
      td.getAttribute("role") === "button");
  }

  function detect() {
    const root = document.getElementById("rc-imageselect");
    if (!root || !CS.isVisible(root)) return null;
    const table = root.querySelector("table[class*='rc-imageselect-table-']");
    if (!table) return null;
    const m = /rc-imageselect-table-(\d)(\d)/.exec(table.className);
    const rows = m ? Number(m[1]) : 0;
    const cols = m ? Number(m[2]) : 0;
    const cells = cellsOf(table);
    const img = table.querySelector("img.rc-image-tile-33, img.rc-image-tile-44, img.rc-image-tile-42") ||
      table.querySelector("img[class*='rc-image-tile-']");
    return {
      root, table, rows, cols, cells, img,
      question: questionOf(root),
      verify: document.getElementById("recaptcha-verify-button"),
      reload: document.getElementById("recaptcha-reload-button"),
      supported: rows === cols && (rows === 3 || rows === 4) && cells.length === rows * cols,
    };
  }

  const signature = (ch) => `${ch.rows}x${ch.cols}|${ch.question}|${ch.img ? ch.img.src : ""}`;
  const imgReady = (img) => !!img && img.complete && img.naturalWidth > 0;
  const ready = (ch) => !!ch && imgReady(ch.img);

  // Pixels of a challenge picture: the loaded <img> (same-origin, no new request), else a
  // same-origin fetch of its URL from inside the frame.
  async function readImage(img) {
    try {
      return await CS.imageToBase64(img, IMAGE_TIMEOUT_MS);
    } catch (e) {
      if (e.code !== "IMAGE_TAINTED") throw e;
    }
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), IMAGE_TIMEOUT_MS);
    try {
      const res = await fetch(img.src, { credentials: "include", cache: "force-cache", signal: ctrl.signal });
      if (!res.ok) throw err("IMAGE_NOT_LOADED", `HTTP ${res.status} for the challenge picture`);
      return await CS.blobToBase64(await res.blob());
    } catch (e) {
      throw e.code ? e : err("IMAGE_NOT_LOADED", `the challenge picture cannot be read (${e.message || e})`);
    } finally {
      clearTimeout(timer);
    }
  }

  async function classify(ch, img, size) {
    const image = await readImage(img);
    return CS.solve(KIND, { image, question: ch.question, size });
  }

  const isSelected = (td) => td.classList.contains("rc-imageselect-tileselected");

  async function clickCell(td) {
    await CS.humanClick(td);
    await CS.sleep(rand(180, 480));
  }

  // Replaced tiles of a dynamic 3x3: classify each new 1x1 picture, click it when it shows
  // the object, until no new picture shows it.
  async function solveDynamic(ch, seen, badge) {
    const end = Date.now() + DYNAMIC_MS;
    let clicks = 0;
    let judged = 0;
    while (Date.now() < end && judged < MAX_DYNAMIC_TILES) {
      // the clicked tiles fade out and get a new picture; wait until every one has it
      const settled = await CS.waitFor(() => !ch.table.querySelector(".rc-imageselect-dynamic-selected") &&
        [...ch.table.querySelectorAll("img.rc-image-tile-11")].every(imgReady), 15000, 200);
      if (!settled) throw err("DYNAMIC_TIMEOUT", "new tiles did not load");
      await CS.sleep(rand(250, 500));
      const fresh = cellsOf(ch.table).map((td, i) => ({ td, i, img: td.querySelector("img.rc-image-tile-11") }))
        .filter((c) => c.img && !seen.has(c.img.src));
      if (!fresh.length) break;
      let picked = 0;
      for (const c of fresh) {
        seen.add(c.img.src);
        judged += 1;
        badge.set("solving", `reCAPTCHA: checking new tile ${c.i + 1}`, false);
        const res = await classify(ch, c.img, 1);
        if (!res.ok) {
          if (res.code === "ERROR_CAPTCHA_UNSOLVABLE") continue; // not sure: leave it
          throw err(res.code || "ERROR", res.message);
        }
        const sol = res.solution || {};
        if (sol.type !== "single" || typeof sol.hasObject !== "boolean") throw err("BAD_SOLUTION", "no hasObject in the answer");
        if (sol.hasObject && c.td.isConnected) {
          await clickCell(c.td);
          picked += 1;
          clicks += 1;
        }
      }
      if (!picked) break;
      await CS.sleep(rand(400, 800));
    }
    return clicks;
  }

  // Answer one picture. Returns {action: "verified", detail} or {action: "reload", detail}.
  async function answer(ch, badge) {
    const sig = signature(ch);
    const label = `reCAPTCHA ${ch.rows}x${ch.cols} “${ch.question}”`;
    badge.set("solving", `${label}: solving`, false);
    const res = await classify(ch, ch.img, ch.rows);
    if (!res.ok) {
      // "unsupported question" / no answer: ask reCAPTCHA for another picture
      if (res.code === "ERROR_CAPTCHA_UNSOLVABLE") return { action: "reload", detail: res.message };
      throw err(res.code || "ERROR", res.message);
    }
    const now = detect();
    if (!now || signature(now) !== sig) return { action: "changed", detail: "the picture changed" };
    const sol = res.solution || {};
    const n = ch.rows * ch.cols;
    if (sol.type !== "multi" || !Array.isArray(sol.objects) || sol.size !== ch.rows) {
      throw err("BAD_SOLUTION", "the answer does not fit the grid");
    }
    const want = new Set(sol.objects.map(Number).filter((i) => Number.isInteger(i) && i >= 0 && i < n));
    const cells = cellsOf(ch.table);
    // toggle only what differs (tiles the user already picked stay as they are when wanted)
    for (let i = 0; i < n; i++) {
      if (want.has(i) !== isSelected(cells[i]) && !cells[i].classList.contains("rc-imageselect-dynamic-selected")) {
        await clickCell(cells[i]);
      }
    }
    let extra = 0;
    if (want.size && ch.rows === 3) {
      await CS.sleep(rand(500, 800));
      const dynamic = cellsOf(ch.table).some((td) => td.classList.contains("rc-imageselect-dynamic-selected") ||
        td.querySelector("img.rc-image-tile-11"));
      if (dynamic) extra = await solveDynamic(ch, new Set([ch.img.src]), badge);
    }
    const detail = want.size ? `picked ${[...want].join(",")}${extra ? ` + ${extra} new tile(s)` : ""}` : "none (skip)";
    const errBefore = visibleError(); // a message still shown from the previous picture is not a verdict
    const at = Date.now();
    await pressVerify(ch);
    return { action: "verified", detail, errBefore, at };
  }

  async function pressVerify(ch) {
    await CS.sleep(rand(350, 800));
    const btn = document.getElementById("recaptcha-verify-button") || ch.verify;
    if (!btn || !CS.isVisible(btn)) throw err("WIDGET_NOT_FOUND", "no Verify button");
    await CS.humanClick(btn);
  }

  async function pressReload(ch) {
    const btn = document.getElementById("recaptcha-reload-button") || ch.reload;
    if (!btn || !CS.isVisible(btn)) throw err("WIDGET_NOT_FOUND", "no reload button");
    await CS.sleep(rand(400, 900));
    await CS.humanClick(btn);
  }

  // A different, loaded picture after `sig`, or null.
  function waitForChange(sig, ms) {
    return CS.waitFor(() => {
      const now = detect();
      return now && signature(now) !== sig && ready(now) ? now : null;
    }, ms, 200);
  }

  // After Verify: "ok" (the checkbox got ticked, or the challenge closed), {next, error} (a new
  // picture, after a wrong answer when reCAPTCHA shows an error), {same, error} or "none".
  // A closed challenge cannot always be seen from inside (Chrome stops updating hidden
  // cross-origin frames), so the checkbox frame's "ticked" message is the main signal.
  async function verdict(sig, errBefore, since) {
    const end = Date.now() + VERDICT_MS;
    const solved = () => state.checkedAt >= since || !state.frameVisible;
    while (Date.now() < end) {
      await CS.sleep(250);
      if (solved()) return { kind: "ok" };
      const now = detect();
      const error = visibleError();
      if (now && signature(now) !== sig) {
        await CS.sleep(300); // the error line is drawn right after the new picture
        return { kind: "next", error: visibleError() };
      }
      if (error && error !== errBefore) return { kind: "same", error };
      if (!now) {
        await CS.sleep(600);
        if (solved()) return { kind: "ok" };
      }
    }
    return solved() ? { kind: "ok" } : { kind: "none" };
  }

  function badgeFor(ch) {
    if (!state.badge) state.badge = new CS.Badge(() => session(true));
    state.badge.attach(ch.table, [ch.verify, ch.reload].filter(Boolean));
    return state.badge;
  }

  async function session(manual) {
    if (state.busy) return;
    const s = CS.settings;
    if (!s.enabled || !s.solvers.recaptcha || !state.frameVisible) return;
    let ch = detect();
    if (!ch) return;
    state.busy = true;
    if (manual) state.gaveUp = false;
    const started = Date.now();
    const badge = badgeFor(ch);
    let rounds = 0;
    let challenges = 0;
    let last = "";
    try {
      while (rounds < MAX_ROUNDS && challenges < MAX_CHALLENGES) {
        if (!state.frameVisible) break;
        ch = ready(ch) && ch.table.isConnected ? ch : await CS.waitFor(() => { const c = detect(); return ready(c) ? c : null; },
          IMAGE_TIMEOUT_MS, 200);
        if (!ch) throw err("WIDGET_NOT_FOUND", "the challenge picture did not load");
        const sig = signature(ch);
        challenges += 1;
        let step;
        if (!ch.supported || !ch.question) {
          step = { action: "reload", detail: `unsupported layout ${ch.rows}x${ch.cols}` };
        } else {
          step = await answer(ch, badge);
        }
        last = step.detail;
        if (step.action === "changed") { ch = null; continue; } // counts as a picture: bounded
        if (step.action === "reload") {
          rounds += 1;
          badge.set("solving", `reCAPTCHA: ${step.detail} - new picture`, false);
          await pressReload(ch);
          ch = await waitForChange(sig, IMAGE_TIMEOUT_MS);
          if (!ch) throw err("WIDGET_NOT_FOUND", "reload gave no new picture");
          continue;
        }
        badge.set("solving", `reCAPTCHA: ${step.detail} - verifying`, false);
        let v = await verdict(sig, step.errBefore || null, step.at);
        if (v.kind === "same" && v.error === "dynamic-more") {
          // reCAPTCHA wants the new tiles looked at once more
          rounds += 1;
          await solveDynamic(ch, new Set([ch.img.src]), badge);
          const at = Date.now();
          await pressVerify(ch);
          v = await verdict(sig, "dynamic-more", at);
        }
        if (v.kind === "ok") {
          state.done.add(sig);
          badge.set("ok", `reCAPTCHA: solved (${challenges} picture(s))`, false);
          CS.outcome(KIND, true, `reCAPTCHA: ${last}, ${challenges} picture(s), ${rounds} retry(ies)`, Date.now() - started);
          return;
        }
        if (v.kind === "next") {
          if (v.error) rounds += 1; // the previous answer was wrong
          badge.set("solving", v.error ? `reCAPTCHA: ${v.error}, next picture` : "reCAPTCHA: next picture", false);
          ch = null;
          continue;
        }
        // same picture with an error, or no reaction at all: take another picture
        rounds += 1;
        badge.set("solving", `reCAPTCHA: ${v.error || "no reaction"}, new picture`, false);
        await pressReload(ch);
        ch = await waitForChange(sig, IMAGE_TIMEOUT_MS);
        if (!ch) throw err("WIDGET_NOT_FOUND", "reload gave no new picture");
      }
      if (!state.frameVisible) return; // closed by the page or the user mid-way
      state.gaveUp = true;
      const why = rounds >= MAX_ROUNDS ? `gave up after ${rounds} failed round(s)` : `gave up after ${challenges} pictures`;
      badge.set("error", `reCAPTCHA: ${why}; press Solve to try again`, true);
      CS.outcome(KIND, false, `reCAPTCHA: ${why} (${last})`, Date.now() - started);
    } catch (e) {
      state.gaveUp = true;
      const detail = `${e.code || "ERROR"}: ${e.message}`;
      badge.set("error", `reCAPTCHA: ${detail}`, true);
      // the background already logged service errors; page-side failures are logged here
      if (!/^ERROR_|^TIMEOUT|^NETWORK|^HTTP|^PERMISSION|^NOT_CONFIGURED|^DISABLED|^BAD_TASK/.test(e.code || "")) {
        CS.outcome(KIND, false, `reCAPTCHA: ${detail}`, Date.now() - started);
      }
    } finally {
      state.busy = false;
    }
  }

  let scanning = false;

  async function scanBframe() {
    if (state.busy || scanning) return;
    scanning = true;
    try {
      await CS.loadSettings();
      const s = CS.settings;
      if (!state.frameVisible) return;
      const ch = detect();
      if (!ch || state.done.has(signature(ch))) return;
      if (!s.enabled || !s.solvers.recaptcha) {
        if (state.badge) state.badge.set("idle", s.enabled ? "reCAPTCHA solver is off" : "Captchosa is off", false);
        return;
      }
      if (state.gaveUp) return; // the pill shows the error and the Solve button
      const badge = badgeFor(ch);
      if (!s.autoSolve) {
        badge.set("idle", "reCAPTCHA: ready", true);
        return;
      }
      if (!ready(ch)) {
        badge.set("solving", "reCAPTCHA: waiting for the picture", false);
        return;
      }
      badge.set("solving", `reCAPTCHA: waiting ${s.delayMs} ms`, false);
      await CS.sleep(s.delayMs);
      if (state.busy || !state.frameVisible || !detect()) return;
      await session(false);
    } finally {
      scanning = false;
    }
  }

  // ------------------------------------------------------------ checkbox frame (anchor)

  // Tell the challenge frame (through the background) that the checkbox got ticked: that is
  // reCAPTCHA's own "solved" signal, which the challenge frame cannot see itself.
  function reportChecked() {
    const box = document.getElementById("recaptcha-anchor");
    const checked = !!box && box.getAttribute("aria-checked") === "true";
    if (checked && !state.anchorChecked) {
      try {
        chrome.runtime.sendMessage({ type: "recaptchaChecked" }, () => void chrome.runtime.lastError);
      } catch (_) {
        // the extension was reloaded: this content script is orphaned
      }
    }
    state.anchorChecked = checked;
  }

  // With auto-solve, tick "I'm not a robot" once (and once more after it expired), like a user.
  async function scanAnchor() {
    reportChecked();
    if (scanning) return;
    scanning = true;
    try {
      await CS.loadSettings();
      const s = CS.settings;
      if (!s.enabled || !s.solvers.recaptcha || !s.autoSolve || !state.frameVisible) return;
      const box = document.getElementById("recaptcha-anchor");
      if (!box || !CS.isVisible(box)) return;
      if (box.getAttribute("aria-checked") === "true" || box.getAttribute("aria-disabled") === "true") return;
      if (/recaptcha-checkbox-(loading|checked|checkmark)/.test(box.className)) return;
      const expired = box.classList.contains("recaptcha-checkbox-expired");
      if (state.anchorClicks >= 2 || (state.anchorClicks === 1 && !expired)) return;
      state.anchorClicks += 1;
      await CS.sleep(s.delayMs + rand(150, 600));
      if (!box.isConnected || box.getAttribute("aria-checked") === "true") return;
      await CS.humanClick(box);
    } finally {
      scanning = false;
    }
  }

  // ------------------------------------------------------------ wiring

  const scan = IS_BFRAME ? scanBframe : scanAnchor;
  let timer = null;
  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(scan, 300);
  }

  function start() {
    if (IS_BFRAME) {
      chrome.runtime.onMessage.addListener((msg, sender) => {
        if (sender.id === chrome.runtime.id && msg && msg.type === "recaptchaChecked") state.checkedAt = Date.now();
        return false;
      });
    }
    watchVisibility((visible) => {
      if (IS_BFRAME && visible && !state.busy) state.gaveUp = false; // a newly opened challenge
      schedule();
    });
    new MutationObserver(schedule).observe(document.documentElement, {
      childList: true, subtree: true, attributes: true, attributeFilter: ["class", "style", "src", "aria-checked"],
    });
    setInterval(() => { if (!state.busy) scan(); }, 2000);
    schedule();
  }

  CS.onSettings((s) => {
    if (state.badge) state.badge.applySettings();
    if (s.enabled && s.solvers.recaptcha && state.gaveUp && !state.busy) state.gaveUp = false;
    schedule();
  });

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
  else start();
})();
