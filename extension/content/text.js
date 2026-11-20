// Image-to-text captchas, two solvers sharing one engine:
//  - blsText: BLS Poland login (5 digits on red curves, input "captcha_code"), module bls_poland
//  - text:    any image captcha next to a captcha input (off by default), module common
// The image is read from the already loaded <img> - never downloaded again,
// because a second request would make the server issue a different captcha.
(function () {
  "use strict";
  const CS = globalThis.CS;
  if (!CS || globalThis.__captchosaText) return;
  globalThis.__captchosaText = true;

  const CAPTCHA_ATTR = /captcha/i;
  const GENERIC_INPUT = /captcha|security.?code|verification.?code|verify.?code|check.?code|kod/i;
  const TEXT_TYPES = new Set(["", "text", "tel", "number", "search"]);

  function attrs(el) {
    return `${el.getAttribute("name") || ""} ${el.id} ${el.getAttribute("placeholder") || ""} ${el.getAttribute("aria-label") || ""}`;
  }

  const SOLVERS = {
    blsText: {
      findInput() {
        return [...document.querySelectorAll('input[name="captcha_code"], input#captcha_code')].find(CS.isVisible) || null;
      },
      findImage(input) { return nearestImage(input, (img) => CAPTCHA_ATTR.test(imgAttrs(img))); },
    },
    text: {
      findInput() {
        return [...document.querySelectorAll("input")].find((i) =>
          TEXT_TYPES.has((i.getAttribute("type") || "").toLowerCase()) && !/^captcha_code$/i.test(i.name) &&
          GENERIC_INPUT.test(attrs(i)) && !/recaptcha|hcaptcha|turnstile/i.test(attrs(i)) && CS.isVisible(i)) || null;
      },
      findImage(input) {
        const byName = nearestImage(input, (img) => CAPTCHA_ATTR.test(imgAttrs(img)));
        if (byName) return byName;
        // no "captcha" in the image attributes: a small image in the same form
        const scope = input.form || input.parentElement;
        return scope ? nearestImage(input, (img) => scope.contains(img) && img.naturalWidth >= 50 &&
          img.naturalWidth <= 400 && img.naturalHeight >= 18 && img.naturalHeight <= 160) : null;
      },
    },
  };

  function imgAttrs(img) {
    return `${img.getAttribute("src") || ""} ${img.id} ${img.className} ${img.alt}`;
  }

  function nearestImage(input, accept) {
    const imgs = [...document.querySelectorAll("img")].filter((img) => CS.isVisible(img) && accept(img));
    if (imgs.length <= 1) return imgs[0] || null;
    const ir = input.getBoundingClientRect();
    const dist = (img) => {
      const r = img.getBoundingClientRect();
      return Math.hypot(r.left - ir.left, r.top - ir.top);
    };
    return imgs.sort((a, b) => dist(a) - dist(b))[0];
  }

  function makeSolver(KIND, finder) {
    const state = { done: new Set(), current: null, badge: null, busy: false, filled: "", tries: new Map() };
    const findInput = finder.findInput;
    const findImage = finder.findImage;

    const keyOf = (img) => `${img.currentSrc || img.src}|${img.naturalWidth}x${img.naturalHeight}`;

    async function scan() {
      if (state.busy) return;
      await CS.loadSettings();
      const s = CS.settings;
      if (!s.solvers[KIND] && !state.badge) return; // stay invisible until switched on
      const input = findInput();
      if (!input) return;
      const img = findImage(input);
      if (!img) return;
      if (!img.complete || !img.naturalWidth) return; // the load event triggers another scan
      const key = keyOf(img);
      if (state.done.has(key)) return;
      if (!state.badge) state.badge = new CS.Badge(() => solve(true));
      state.badge.attach(img);
      state.current = { key, img, input };
      if (!s.enabled || !s.solvers.blsText) {
        state.badge.set("idle", s.enabled ? "solver is off" : "Captchosa is off", false);
        return;
      }
      if ((state.tries.get(key) || 0) >= s.retries) {
        state.badge.set("error", `gave up after ${s.retries} tries; press Solve to retry`, true);
        return;
      }
      if (!s.autoSolve) {
        state.badge.set("idle", "text captcha: ready", true);
        return;
      }
      state.badge.set("solving", `waiting ${s.delayMs} ms`, false);
      await CS.sleep(s.delayMs);
      if (!state.current || state.current.key !== key) return;
      await solve(false);
    }

    async function solve(manual) {
      if (state.busy || !state.current) return;
      const s = CS.settings;
      if (!s.enabled || !s.solvers.blsText) return;
      const { key, img, input } = state.current;
      if (manual) state.tries.set(key, 0);
      state.tries.set(key, (state.tries.get(key) || 0) + 1);
      state.busy = true;
      const started = Date.now();
      try {
        state.badge.set("solving", "reading the captcha", false);
        let body;
        try {
          body = await CS.imageToBase64(img);
        } catch (err) {
          const detail = `${err.code || "IMAGE_ERROR"}: ${err.message}`;
          state.badge.set("error", detail, true);
          CS.outcome(KIND, false, detail, Date.now() - started);
          return;
        }
        const res = await CS.solve(KIND, { body });
        if (!res.ok) {
          state.badge.set("error", `${res.code}: ${res.message}`, true);
          setTimeout(scan, 1000); // retry while tries are left
          return;
        }
        const text = String((res.solution && res.solution.text) || "").trim();
        if (!/^[0-9A-Za-z]{1,12}$/.test(text)) {
          state.badge.set("error", "BAD_SOLUTION: empty or odd text", true);
          CS.outcome(KIND, false, "BAD_SOLUTION: empty or odd text", Date.now() - started);
          return;
        }
        if (keyOf(img) !== key) return; // the captcha changed meanwhile
        if (input.value && input.value !== state.filled) {
          state.badge.set("idle", `read ${text} - the field already has your input`, false);
          state.done.add(key);
          return;
        }
        CS.setInputValue(input, text);
        state.filled = text;
        state.done.add(key);
        let submitted = false;
        if (s.autoSolve && s.submitAfterFill && input.form) {
          await CS.sleep(300 + Math.random() * 300);
          const btn = [...input.form.querySelectorAll('button[type=submit], input[type=submit], button:not([type])')]
            .find(CS.isVisible);
          if (btn) await CS.humanClick(btn);
          else input.form.requestSubmit();
          submitted = true;
        }
        const conf = res.solution && typeof res.solution.confidence === "number"
          ? ` (${Math.round(res.solution.confidence * 100)}%)` : "";
        const detail = `filled ${text}${conf}${submitted ? " - submitted" : ""}`;
        state.badge.set("ok", detail, false);
        CS.outcome(KIND, true, detail, Date.now() - started);
      } finally {
        state.busy = false;
      }
    }

    let timer = null;
    const schedule = () => {
      clearTimeout(timer);
      timer = setTimeout(scan, 300);
    };

    function start() {
      new MutationObserver(schedule).observe(document.documentElement, {
        childList: true, subtree: true, attributes: true, attributeFilter: ["src"],
      });
      document.addEventListener("load", (e) => { if (e.target && e.target.tagName === "IMG") schedule(); }, true);
      setInterval(scan, 2500);
      schedule();
    }

    CS.onSettings(() => {
      if (state.badge) state.badge.applySettings();
      schedule();
    });

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
    else start();
  }

  for (const [kind, finder] of Object.entries(SOLVERS)) makeSolver(kind, finder);
})();
