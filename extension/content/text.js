// Image-to-text captchas, two solvers sharing one engine:
// One "Text captcha" solver (kind "text", one switch) with two finders:
//  - BLS Poland login (5 digits on red curves, input "captcha_code") -> module bls_poland
//  - any image captcha next to a text input                         -> module common
//
// Detection (generic solver): a captcha is a pair (picture, text input).
//  * pictures: <img>, <canvas>, <input type=image> and elements with a CSS background image,
//    also inside open shadow roots; captcha-like by attributes (src/id/class/alt...) or by
//    size/shape (wide and short) when they sit next to a captcha-like input;
//  * inputs: text-like inputs whose attributes, label or surrounding text talk about a
//    captcha / security / verification code (several languages).
// The picture is read from the page (never downloaded again: a second request would make the
// server issue a different captcha). Cross-origin pictures are cut out of a screenshot of the
// visible tab by the service worker instead.
// While solving, the input shows "Solving…"; the answer is typed into the input.
(function () {
  "use strict";
  const CS = globalThis.CS;
  if (!CS || globalThis.__captchosaText) return;
  globalThis.__captchosaText = true;

  const CAPTCHA_ATTR = /captcha/i;
  const IMG_HINT = /captcha|captch|kaptcha|securimage|botdetect|verif|vcode|valid(ate|ation)?.?(code|img|image)|check.?code|sec.?code|auth.?code|rand.?code|img.?code|image.?code|get.?code|code.?img|code.?image|antibot|kod|code\.(php|aspx?|ashx|jsp|do|cgi)/i;
  const INPUT_HINT = /captcha|captch|kaptcha|security.?code|verif(y|ication)?.?code|vcode|valid(ate|ation)?.?code|check.?code|auth.?code|img.?code|image.?code|rand.?code|sec.?code|antibot|\bkod|kodu|код|doğrulama|dogrulama|güvenlik|guvenlik|yoxlama/i;
  const TEXT_HINT = /captcha|security code|verification code|enter the (code|text|characters|word)|type the (code|text|characters)|characters (shown|above|below|you see|in the (image|picture))|code (shown|above|below|in the (image|picture))|text (shown|in the (image|picture))|what (code|text) is in the image|resimdeki|güvenlik kodu|doğrulama kodu|kodu (giriniz|girin|daxil)|şəkildəki|təhlükəsizlik kodu|yoxlama kodu|код с картинки|введите (код|символы|текст)|проверочный код|символы с картинки/i;
  const NOT_INPUT = /recaptcha|hcaptcha|turnstile|e-?mail|user.?name|login|passw|phone|mobile|search|query|\botp\b|sms|zip|postal|card|cvv|cvc|coupon|promo|amount|price|quantity|qty|date|year|month/i;
  const TEXT_TYPES = new Set(["", "text", "tel", "number", "search"]);
  const MAX_PAIR_DIST = 450;   // px between a captcha picture and its input
  const solvingLabel = () => {
    try { return chrome.i18n.getMessage("solvingPlaceholder") || "Solving…"; } catch (_) { return "Solving…"; }
  };

  function attrs(el) {
    return ["name", "id", "placeholder", "aria-label", "title", "class", "autocomplete", "data-name"]
      .map((a) => el.getAttribute(a) || "").join(" ");
  }

  function imgAttrs(el) {
    const own = ["src", "id", "class", "alt", "title", "name", "data-src", "aria-label"]
      .map((a) => el.getAttribute(a) || "").join(" ");
    const parent = el.parentElement;
    const p = parent ? `${parent.id} ${parent.className && parent.className.baseVal === undefined ? parent.className : ""} ${parent.getAttribute("href") || ""}` : "";
    let bg = "";
    if (el.tagName !== "IMG" && el.tagName !== "CANVAS" && el.tagName !== "INPUT") {
      bg = getComputedStyle(el).backgroundImage || "";
    }
    return `${own} ${p} ${bg}`;
  }

  // Text that describes an input: its labels, aria-labelledby and the text around it.
  function describedBy(input) {
    const parts = [];
    try { for (const l of input.labels || []) parts.push(l.textContent || ""); } catch (_) { /* detached */ }
    for (const id of (input.getAttribute("aria-labelledby") || "").split(/\s+/)) {
      const el = id && input.getRootNode().getElementById ? input.getRootNode().getElementById(id) : null;
      if (el) parts.push(el.textContent || "");
    }
    let node = input.parentElement;
    for (let depth = 0; node && depth < 3; depth += 1, node = node.parentElement) {
      const t = (node.innerText || node.textContent || "").trim();
      if (t.length > 400) break;
      parts.push(t);
    }
    return parts.join(" ").replace(/\s+/g, " ").slice(0, 1200);
  }

  function center(el) {
    const r = el.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2, r };
  }

  // Shortest distance between two boxes (0 when they overlap).
  function boxDist(a, b) {
    const ra = a.getBoundingClientRect();
    const rb = b.getBoundingClientRect();
    const dx = Math.max(0, ra.left - rb.right, rb.left - ra.right);
    const dy = Math.max(0, ra.top - rb.bottom, rb.top - ra.bottom);
    return Math.hypot(dx, dy);
  }

  function pictureSize(el) {
    if (el.tagName === "IMG") return { w: el.naturalWidth || el.width, h: el.naturalHeight || el.height };
    if (el.tagName === "CANVAS") return { w: el.width, h: el.height };
    const r = el.getBoundingClientRect();
    return { w: r.width, h: r.height };
  }

  // Wide, short pictures of plausible size look like text captchas.
  function captchaShaped(el) {
    const { w, h } = pictureSize(el);
    const r = el.getBoundingClientRect();
    const W = Math.max(w, r.width);
    const H = Math.max(h, r.height);
    return W >= 40 && W <= 600 && H >= 15 && H <= 220 && W / H >= 1.1 && W / H <= 12;
  }

  function textInputs() {
    return CS.deepQueryAll("input, textarea").filter((i) =>
      (i.tagName === "TEXTAREA" ? i.rows <= 2 : TEXT_TYPES.has((i.getAttribute("type") || "").toLowerCase())) &&
      !i.disabled && !i.readOnly && CS.isVisible(i));
  }

  function pictures() {
    const out = CS.deepQueryAll("img, canvas, input[type=image]").filter(CS.isVisible);
    // elements painted with a background image whose URL looks like a captcha
    for (const el of CS.deepQueryAll("div, span, a, i, label, button")) {
      if (el.children.length > 2) continue;
      const bg = el.style.backgroundImage || "";
      if (bg && IMG_HINT.test(bg) && CS.isVisible(el)) out.push(el);
    }
    return out;
  }

  // Generic finder: returns {input, img} or null.
  function findGenericPair() {
    const inputs = textInputs().filter((i) => !/^captcha_code$/i.test(i.name || i.id));
    if (!inputs.length) return null;
    const pics = pictures();
    if (!pics.length) return null;

    const scored = [];
    for (const input of inputs) {
      const a = attrs(input);
      if (/recaptcha|hcaptcha|turnstile/i.test(a)) continue;
      let s = 0;
      if (INPUT_HINT.test(a)) s += 3;
      else if (NOT_INPUT.test(a)) continue;
      if (TEXT_HINT.test(describedBy(input))) s += 2;
      const ml = Number.parseInt(input.getAttribute("maxlength") || "", 10);
      if (ml >= 3 && ml <= 10) s += 0.5;
      if ((input.getAttribute("autocomplete") || "").toLowerCase() === "off") s += 0.25;
      scored.push({ input, s });
    }

    let best = null;
    for (const { input, s: is } of scored) {
      for (const pic of pics) {
        const d = boxDist(input, pic);
        if (d > MAX_PAIR_DIST) continue;
        const hinted = IMG_HINT.test(imgAttrs(pic));
        const shaped = captchaShaped(pic);
        if (!hinted && !shaped) continue;
        const sameForm = !!(input.form && input.form.contains(pic));
        // need evidence on at least one side, and some on the other
        const evidence = (is >= 2 ? 1 : 0) + (hinted ? 1 : 0);
        if (evidence === 0) continue;
        if (!hinted && !(shaped && (sameForm || d <= 250))) continue;
        if (is < 0.5 && !(hinted && (sameForm || d <= 250))) continue;
        const score = is + (hinted ? 3 : 0) + (shaped ? 1 : 0) + (sameForm ? 2 : 0) - d / 150;
        if (!best || score > best.score) best = { input, img: pic, score };
      }
    }
    return best;
  }

  // finder name -> [finder, ImageToTextTask module]
  const SOLVERS = {
    blsPoland: {
      module: "bls_poland",
      find() {
        const input = [...document.querySelectorAll('input[name="captcha_code"], input#captcha_code')]
          .find(CS.isVisible);
        if (!input) return null;
        const imgs = [...document.querySelectorAll("img")].filter((img) => CS.isVisible(img) &&
          CAPTCHA_ATTR.test(imgAttrs(img)));
        if (!imgs.length) return null;
        imgs.sort((a, b) => boxDist(a, input) - boxDist(b, input));
        return { input, img: imgs[0] };
      },
    },
    general: { module: "common", find: findGenericPair },
  };

  // ---- reading the picture

  function hashString(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i += Math.max(1, Math.floor(str.length / 4096))) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h >>> 0).toString(16) + ":" + str.length;
  }

  function keyOf(el) {
    if (el.tagName === "IMG") return `img|${el.currentSrc || el.src}|${el.naturalWidth}x${el.naturalHeight}`;
    if (el.tagName === "CANVAS") {
      try { return `canvas|${hashString(el.toDataURL("image/png"))}`; } catch (_) { return `canvas|tainted|${el.width}x${el.height}`; }
    }
    if (el.tagName === "INPUT") return `input|${el.src}`;
    return `bg|${el.style.backgroundImage || getComputedStyle(el).backgroundImage}`;
  }

  function ready(el) {
    if (el.tagName === "IMG") return el.complete && el.naturalWidth > 0;
    if (el.tagName === "CANVAS") return el.width > 0 && el.height > 0;
    return true;
  }

  // Cut the element out of a screenshot of the visible tab (service worker).
  async function captureElement(el) {
    if (window.top !== window) {
      throw Object.assign(new Error("cross-origin captcha inside a frame cannot be read"), { code: "IMAGE_TAINTED" });
    }
    const r0 = el.getBoundingClientRect();
    if (r0.top < 0 || r0.left < 0 || r0.bottom > innerHeight || r0.right > innerWidth) {
      el.scrollIntoView({ block: "center", inline: "nearest" });
      await CS.sleep(250);
    }
    const r = el.getBoundingClientRect();
    const res = await CS.capture({ x: r.left, y: r.top, w: r.width, h: r.height, vw: innerWidth });
    if (!res.ok) throw Object.assign(new Error(res.message), { code: res.code || "CAPTURE_FAILED" });
    return res.body;
  }

  async function pictureToBase64(el) {
    try {
      if (el.tagName === "IMG") return await CS.imageToBase64(el);
      if (el.tagName === "CANVAS") {
        try {
          return el.toDataURL("image/png").split(",", 2)[1];
        } catch (_) {
          throw Object.assign(new Error("tainted canvas"), { code: "IMAGE_TAINTED" });
        }
      }
    } catch (err) {
      if (err.code !== "IMAGE_TAINTED") throw err;
    }
    return captureElement(el);
  }

  // ---- "Solving…" shown inside the input while the answer is on its way

  function showSolving(input) {
    if (input.value) return () => {};
    const had = input.hasAttribute("placeholder");
    const prev = input.getAttribute("placeholder");
    input.setAttribute("placeholder", solvingLabel());
    return () => {
      if (input.getAttribute("placeholder") !== solvingLabel()) return; // the page changed it
      if (had) input.setAttribute("placeholder", prev);
      else input.removeAttribute("placeholder");
    };
  }

  function makeSolver(finder) {
    const KIND = "text";
    const state = { done: new Set(), current: null, badge: null, busy: false, filled: "", tries: new Map() };

    async function scan() {
      if (state.busy) return;
      await CS.loadSettings();
      const s = CS.settings;
      if (!s.solvers[KIND] && !state.badge) return; // stay invisible until switched on
      const pair = finder.find();
      if (!pair) return;
      const { input, img } = pair;
      if (!ready(img)) return; // the load event triggers another scan
      const key = keyOf(img);
      if (state.done.has(key)) return;
      if (!state.badge) state.badge = new CS.Badge(() => solve(true));
      state.badge.attach(img, [input]);
      state.current = { key, img, input };
      if (!s.enabled || !s.solvers[KIND]) {
        state.badge.set("idle", s.enabled ? "solver is off" : "Captchosa is off", false);
        return;
      }
      if ((state.tries.get(key) || 0) >= s.retries) {
        state.badge.set("error", `gave up after ${s.retries} tries; press Solve to retry`, true);
        return;
      }
      if (!s.autoSolve) {
        state.badge.set("idle", "text captcha detected", true);
        return;
      }
      state.badge.set("solving", solvingLabel(), false);
      await CS.sleep(s.delayMs);
      if (!state.current || state.current.key !== key) return;
      await solve(false);
    }

    async function solve(manual) {
      if (state.busy || !state.current) return;
      const s = CS.settings;
      if (!s.enabled || !s.solvers[KIND]) return;
      const { key, img, input } = state.current;
      if (manual) state.tries.set(key, 0);
      state.tries.set(key, (state.tries.get(key) || 0) + 1);
      state.busy = true;
      const started = Date.now();
      const restore = showSolving(input);
      try {
        state.badge.set("solving", solvingLabel(), false);
        let body;
        try {
          body = await pictureToBase64(img);
        } catch (err) {
          const detail = `${err.code || "IMAGE_ERROR"}: ${err.message}`;
          state.badge.set("error", detail, true);
          CS.outcome(KIND, false, detail, Date.now() - started);
          return;
        }
        const res = await CS.solve(KIND, { body, module: finder.module });
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
        restore();
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
        restore();
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
        childList: true, subtree: true, attributes: true, attributeFilter: ["src", "style", "class"],
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

  // exposed for the end-to-end tests
  globalThis.__captchosaFindText = () => {
    const p = findGenericPair();
    return p ? { input: p.input.id || p.input.name || "", img: p.img.id || p.img.tagName } : null;
  };

  for (const finder of Object.values(SOLVERS)) makeSolver(finder);
})();
