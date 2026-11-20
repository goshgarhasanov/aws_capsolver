// BLS grid captcha ("select all boxes with number 748"), e.g. on BLS Spain visa
// pages. Only the visible tiles count: BLS renders many decoy tiles and labels
// hidden by CSS, off-screen positions or other tiles stacked above them.
(function () {
  "use strict";
  const CS = globalThis.CS;
  if (!CS || globalThis.__captchosaBls) return;
  globalThis.__captchosaBls = true;

  const KIND = "blsGrid";
  const state = { handled: new Set(), done: new Set(), attempts: 0, current: null, badge: null, busy: false };

  // Bring the real grid on screen so the hit-tests below can see it.
  function scrollGridIntoView() {
    const first = [...document.querySelectorAll("img.captcha-img")].find(CS.isVisible);
    if (!first) return;
    const r = first.getBoundingClientRect();
    if (r.top < 0 || r.bottom > innerHeight) first.scrollIntoView({ block: "center", inline: "nearest" });
  }

  function extract() {
    const labels = [...document.querySelectorAll(".box-label")];
    const tilesAll = [...document.querySelectorAll("img.captcha-img")];
    if (!labels.length || tilesAll.length < 9) return null;
    scrollGridIntoView();
    let target = null;
    let fallback = null;
    for (const lbl of labels) {
      if (!CS.isVisible(lbl)) continue;
      const m = /(\d{3})/.exec(lbl.textContent || "");
      if (!m) continue;
      if (fallback === null) fallback = m[1];
      if (CS.hitTest(lbl)) { target = m[1]; break; }
    }
    target = target || fallback;
    // page order = document order of the visible, uncovered tiles
    const tiles = tilesAll.filter((img) => CS.isVisible(img) && CS.hitTest(img));
    if (!target || tiles.length !== 9) return { target, tiles, incomplete: true };
    return { target, tiles };
  }

  function signature(found) {
    return `${found.target}|` + found.tiles.map((t) => {
      const s = t.getAttribute("src") || "";
      return `${s.length}:${s.slice(-24)}`;
    }).join(",");
  }

  function rawKey() {
    const imgs = document.querySelectorAll("img.captcha-img");
    if (imgs.length < 9) return "";
    let key = "";
    for (const t of imgs) {
      const s = t.getAttribute("src") || "";
      key += `${s.length}:${s.slice(-16)},`;
    }
    return key;
  }

  function anchorFor(tiles) {
    const a = tiles[0];
    return a.closest("form") || (a.parentElement && a.parentElement.parentElement) || a;
  }

  function findSubmit(tiles) {
    const form = tiles[0].closest("form");
    const scope = form || document;
    const btn = [...scope.querySelectorAll("#btnVerify, button[type=submit], input[type=submit]")]
      .find((b) => CS.isVisible(b) && !b.disabled);
    return btn || [...document.querySelectorAll("#btnVerify")].find(CS.isVisible) || null;
  }

  function emptyPassword() {
    return [...document.querySelectorAll("input[type=password]")].some((i) => CS.isVisible(i) && !i.value);
  }

  async function scan() {
    if (state.busy) return;
    await CS.loadSettings();
    const s = CS.settings;
    if (!s.enabled || !s.solvers.blsGrid) {
      if (state.badge) state.badge.set("idle", s.enabled ? "BLS grid solver is off" : "Captchosa is off", false);
      return;
    }
    // cheap key over every tile (decoys included): skip grids already answered
    // without scrolling or hit-testing again
    const pageKey = rawKey();
    if (!pageKey || state.done.has(pageKey)) return;
    const found = state.badge ? await state.badge.hidden(extract) : extract();
    if (!found || found.incomplete) return;
    const sig = signature(found);
    if (state.handled.has(sig)) return;
    state.current = { sig, found, pageKey };
    if (!state.badge) state.badge = new CS.Badge(() => solve(true));
    state.badge.attach(anchorFor(found.tiles));
    if (state.attempts >= s.retries) {
      state.badge.set("error", `gave up after ${s.retries} grid(s); press Solve to try again`, true);
      return;
    }
    if (!s.autoSolve) {
      state.badge.set("idle", `BLS grid ${found.target}: ready`, true);
      return;
    }
    state.badge.set("solving", `waiting ${s.delayMs} ms`, false);
    await CS.sleep(s.delayMs);
    if (!state.current || state.current.sig !== sig) return;
    await solve(false);
  }

  async function solve(manual) {
    if (state.busy || !state.current) return;
    const s = CS.settings;
    if (!s.enabled || !s.solvers.blsGrid) return;
    state.busy = true;
    const started = Date.now();
    try {
      // re-read right before solving: the page may have swapped the grid
      const found = await state.badge.hidden(extract);
      if (!found || found.incomplete) {
        state.badge.set("error", "WIDGET_NOT_FOUND: the 9 visible tiles are gone", true);
        return;
      }
      const sig = signature(found);
      state.handled.add(sig);
      state.attempts = manual ? 1 : state.attempts + 1;
      state.badge.set("solving", `BLS grid ${found.target}: solving`, false);
      let images;
      try {
        images = await Promise.all(found.tiles.map((t) => CS.imageToBase64(t)));
      } catch (err) {
        const detail = `${err.code || "IMAGE_ERROR"}: ${err.message}`;
        state.badge.set("error", detail, true);
        CS.outcome(KIND, false, detail, Date.now() - started);
        return;
      }
      const res = await CS.solve(KIND, { target: found.target, images });
      if (!res.ok) {
        state.badge.set("error", `${res.code}: ${res.message}`, true);
        return;
      }
      const objects = (res.solution && res.solution.objects) || [];
      const idx = [...new Set(objects.map(Number))].filter((i) => Number.isInteger(i) && i >= 0 && i < 9);
      if (!idx.length) {
        state.badge.set("error", "BAD_SOLUTION: no tiles to click", true);
        CS.outcome(KIND, false, "BAD_SOLUTION: no tiles", Date.now() - started);
        return;
      }
      for (const i of idx) {
        await CS.humanClick(found.tiles[i]);
        await CS.sleep(120 + Math.random() * 260);
      }
      const detail = `${found.target}: picked ${idx.join(",")}`;
      let submitted = false;
      if (s.autoSolve && s.blsGridSubmit && !emptyPassword()) {
        const btn = findSubmit(found.tiles);
        if (btn) {
          await CS.sleep(250 + Math.random() * 300);
          await CS.humanClick(btn);
          submitted = true;
        }
      }
      const note = submitted ? " - submitted" : (emptyPassword() ? " - type the password and submit" : "");
      state.done.add(state.current.pageKey);
      state.badge.set("ok", detail + note, false);
      CS.outcome(KIND, true, detail + (submitted ? " (submitted)" : ""), Date.now() - started);
    } finally {
      state.busy = false;
    }
  }

  // BLS loads the grid after an AJAX call: watch the DOM (debounced) plus a slow poll.
  let timer = null;
  const schedule = () => {
    clearTimeout(timer);
    timer = setTimeout(scan, 300);
  };

  function start() {
    new MutationObserver(schedule).observe(document.documentElement, {
      childList: true, subtree: true, attributes: true, attributeFilter: ["src", "style", "class"],
    });
    setInterval(scan, 2000);
    schedule();
  }

  CS.onSettings(() => {
    if (state.badge) state.badge.applySettings();
    // a captcha that was not answered yet gets another chance under the new settings
    if (state.current && !state.done.has(state.current.pageKey)) state.handled.delete(state.current.sig);
    schedule();
  });

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
  else start();
})();
