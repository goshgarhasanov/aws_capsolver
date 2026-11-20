// Isolated-world helpers shared by the solver content scripts (aws.js, bls.js, bls_text.js).
// Sets globalThis.CS (captchosa shared).
(function () {
  "use strict";
  if (globalThis.CS) return;

  const { normalize } = globalThis.CAPTCHOSA;
  let settings = normalize(null);
  const listeners = new Set();
  let ready;

  function onSettings(fn) {
    listeners.add(fn);
  }

  function loadSettings() {
    if (!ready) {
      ready = chrome.storage.local.get("settings").then(({ settings: raw }) => {
        settings = normalize(raw);
        return settings;
      }, () => settings);
    }
    return ready;
  }

  // Settings apply live: every change is pushed to the solvers without a reload.
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local" || !changes.settings) return;
    settings = normalize(changes.settings.newValue);
    for (const fn of listeners) {
      try { fn(settings); } catch (err) { console.warn("captchosa:", err); }
    }
  });

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function waitFor(fn, timeoutMs, stepMs) {
    const end = Date.now() + timeoutMs;
    for (;;) {
      const v = fn();
      if (v) return v;
      if (Date.now() >= end) return null;
      await sleep(stepMs || 150);
    }
  }

  function send(msg, timeoutMs) {
    return new Promise((resolve) => {
      const timer = setTimeout(() => resolve({ ok: false, code: "TIMEOUT", message: "the extension did not answer" }),
        timeoutMs || 180000);
      try {
        chrome.runtime.sendMessage(msg, (res) => {
          clearTimeout(timer);
          const err = chrome.runtime.lastError;
          if (err) resolve({ ok: false, code: "EXTENSION_ERROR", message: err.message });
          else resolve(res || { ok: false, code: "EXTENSION_ERROR", message: "empty answer" });
        });
      } catch (err) {
        clearTimeout(timer);
        // the extension was reloaded: this content script is orphaned
        resolve({ ok: false, code: "EXTENSION_ERROR", message: String(err && err.message || err) });
      }
    });
  }

  const solve = (kind, task) => send({ type: "solve", kind, task });
  // Screenshot crop of a page rectangle (CSS px of the top frame) -> {ok, body: base64 PNG}
  const capture = (rect) => send({ type: "capture", rect }, 15000);
  const outcome = (kind, ok, detail, ms) => send({ type: "outcome", kind, ok, detail, ms }, 10000);

  // ---- DOM helpers

  // querySelectorAll across the document and every open shadow root below it.
  function deepQueryAll(selector, root) {
    const out = [];
    const visit = (node) => {
      if (!node || !node.querySelectorAll) return;
      out.push(...node.querySelectorAll(selector));
      const all = node.querySelectorAll("*");
      for (const el of all) if (el.shadowRoot) visit(el.shadowRoot);
    };
    visit(root || document);
    return out;
  }

  function isVisible(el) {
    if (!el || !el.isConnected) return false;
    const s = getComputedStyle(el);
    if (s.display === "none" || s.visibility === "hidden" || parseFloat(s.opacity || "1") === 0) return false;
    if (typeof el.checkVisibility === "function" &&
        !el.checkVisibility({ opacityProperty: true, visibilityProperty: true })) return false;
    const r = el.getBoundingClientRect();
    return r.width > 2 && r.height > 2;
  }

  // True when the element's centre is not covered by another element (decoys are).
  function hitTest(el) {
    const r = el.getBoundingClientRect();
    const x = r.left + r.width / 2;
    const y = r.top + r.height / 2;
    if (x < 0 || y < 0 || x > innerWidth || y > innerHeight) return false;
    const root = el.getRootNode();
    const hit = (root.elementFromPoint ? root : document).elementFromPoint(x, y);
    return !!hit && (hit === el || el.contains(hit) || hit.contains(el));
  }

  const rand = (a, b) => a + Math.random() * (b - a);

  // A human-like click: pointer/mouse sequence at a jittered point inside the element.
  async function humanClick(el, point) {
    const r = el.getBoundingClientRect();
    const x = point ? point.x : r.left + r.width * rand(0.35, 0.65);
    const y = point ? point.y : r.top + r.height * rand(0.35, 0.65);
    const base = { bubbles: true, cancelable: true, composed: true, view: window, clientX: x, clientY: y,
      screenX: x + (window.screenX || 0), screenY: y + (window.screenY || 0), button: 0 };
    const ptr = Object.assign({ pointerId: 1, pointerType: "mouse", isPrimary: true }, base);
    el.dispatchEvent(new PointerEvent("pointerover", ptr));
    el.dispatchEvent(new MouseEvent("mouseover", base));
    el.dispatchEvent(new PointerEvent("pointermove", ptr));
    el.dispatchEvent(new MouseEvent("mousemove", base));
    await sleep(rand(30, 90));
    el.dispatchEvent(new PointerEvent("pointerdown", Object.assign({ buttons: 1 }, ptr)));
    el.dispatchEvent(new MouseEvent("mousedown", Object.assign({ buttons: 1 }, base)));
    if (typeof el.focus === "function") el.focus({ preventScroll: true });
    await sleep(rand(40, 110));
    el.dispatchEvent(new PointerEvent("pointerup", ptr));
    el.dispatchEvent(new MouseEvent("mouseup", base));
    el.dispatchEvent(new MouseEvent("click", Object.assign({ detail: 1 }, base)));
  }

  // Type a value into an input the way frameworks notice it (React/Vue/jQuery).
  function setInputValue(input, value) {
    const proto = input instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
    input.focus({ preventScroll: true });
    setter.call(input, value);
    input.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: value }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    input.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true, key: value.slice(-1) }));
  }

  // ---- in-page status pill + manual Solve button (own shadow root and styles)

  const BADGE_CSS = `
    :host { all: initial; }
    .pill { position: fixed; z-index: 2147483647; display: flex; align-items: center; gap: 7px;
      font: 500 12px/1.3 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: #e2e8f0;
      background: linear-gradient(180deg, #1f2a44, #0b1222); border: 1px solid #ffffff1a;
      border-radius: 999px; padding: 3px 4px 3px 4px; box-shadow: 0 4px 14px #0f172a40;
      pointer-events: auto; cursor: grab; user-select: none; touch-action: none; max-width: 360px;
      transition: opacity .2s ease; }
    .pill.dragging { cursor: grabbing; opacity: .85; }
    .pill.compact { gap: 2px; padding: 2px; max-width: none; }
    .pill.compact .msg { display: none; }
    .pill.compact button.solve { padding: 4px 8px; }
    button.close { background: transparent; color: #94a3b8; padding: 2px 6px; font: 700 14px/1 system-ui, sans-serif;
      border-radius: 999px; }
    button.close:hover { color: #f1f5f9; background: #ffffff1a; filter: none; }
    .logo { width: 20px; height: 20px; flex: none; }
    .logo .ck { stroke: #f59e0b; transition: stroke .2s ease; }
    .solving .logo .ck { stroke: #3b82f6; animation: p 1.1s ease-in-out infinite; }
    .ok .logo .ck { stroke: #22c55e; } .error .logo .ck { stroke: #ef4444; } .idle .logo .ck { stroke: #f59e0b; }
    .msg { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding-right: 6px; }
    button { pointer-events: auto; font: 700 12px/1 system-ui, -apple-system, "Segoe UI", sans-serif; color: #1f1300;
      background: linear-gradient(180deg, #fcd34d, #f59e0b); border: 0; border-radius: 999px; padding: 6px 12px;
      cursor: pointer; transition: filter .15s ease, transform .15s ease; }
    button:hover { filter: brightness(1.06); } button:active { transform: translateY(1px); }
    button:focus-visible { outline: 2px solid #fde68a; outline-offset: 2px; }
    button[hidden] { display: none; }
    @keyframes p { 50% { opacity: .35; } }
    @media (prefers-reduced-motion: reduce) { .pill, button { transition: none; } .solving .logo .ck { animation: none; } }`;

  // The Captchosa mark, inline (no extra request, no web_accessible_resources).
  const MARK = '<svg class="logo" viewBox="0 0 128 128" aria-hidden="true">' +
    '<path d="M83.1 91.6 A36 36 0 1 1 83.1 36.4" fill="none" stroke="#f1f5f9" stroke-width="16" stroke-linecap="round"/>' +
    '<path class="ck" d="M43 63 L58 78 L104 30" fill="none" stroke-width="16" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  function t(key, fallback) {
    try { return chrome.i18n.getMessage(key) || fallback; } catch (_) { return fallback; }
  }

  class Badge {
    constructor(onSolve) {
      this.host = document.createElement("captchosa-badge");
      this.host.style.cssText = "all: initial; position: static;";
      const root = this.host.attachShadow({ mode: "open" }); // open: end-to-end tests click it
      root.innerHTML = `<style>${BADGE_CSS}</style><div class="pill idle" role="status">${MARK}` +
        '<span class="msg">Captchosa</span><button type="button" class="solve" hidden></button>' +
        '<button type="button" class="close">×</button></div>';
      this.pill = root.querySelector(".pill");
      this.msg = root.querySelector(".msg");
      this.button = root.querySelector("button.solve");
      this.button.textContent = t("solveButton", "Solve");
      this.closeBtn = root.querySelector("button.close");
      this.closeBtn.title = t("closeBadge", "Hide");
      this.closeBtn.setAttribute("aria-label", t("closeBadge", "Hide"));
      this.button.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        onSolve();
      });
      // x hides the pill for this captcha (the solver keeps working)
      this.dismissed = false;
      this.closeBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.dismissed = true;
        this.host.style.display = "none";
      });
      // drag anywhere on the pill except its buttons; the spot is kept relative to the captcha
      this.manual = null;
      this.pill.addEventListener("pointerdown", (e) => {
        if (e.button !== 0 || e.target.closest("button")) return;
        e.preventDefault();
        e.stopPropagation();
        const start = { x: e.clientX, y: e.clientY, l: parseFloat(this.pill.style.left) || 0,
          t: parseFloat(this.pill.style.top) || 0 };
        try { this.pill.setPointerCapture(e.pointerId); } catch (_) { /* synthetic pointer */ }
        this.pill.classList.add("dragging");
        const move = (ev) => {
          ev.preventDefault();
          const l = Math.max(0, Math.min(innerWidth - this.pill.offsetWidth, start.l + ev.clientX - start.x));
          const top = Math.max(0, Math.min(innerHeight - this.pill.offsetHeight, start.t + ev.clientY - start.y));
          this.pill.style.left = `${Math.round(l)}px`;
          this.pill.style.top = `${Math.round(top)}px`;
          const r = this.anchor && this.anchor.isConnected ? this.anchor.getBoundingClientRect() : { left: 0, top: 0 };
          this.manual = { dx: l - r.left, dy: top - r.top };
        };
        const up = (ev) => {
          try { this.pill.releasePointerCapture(ev.pointerId); } catch (_) { /* already released */ }
          this.pill.classList.remove("dragging");
          this.pill.removeEventListener("pointermove", move);
          this.pill.removeEventListener("pointerup", up);
          this.pill.removeEventListener("pointercancel", up);
        };
        this.pill.addEventListener("pointermove", move);
        this.pill.addEventListener("pointerup", up);
        this.pill.addEventListener("pointercancel", up);
      });
      this.anchor = null;
      this.reposition = this.reposition.bind(this);
    }

    // anchor: the captcha; avoid: other elements the pill must never cover (e.g. the input)
    attach(anchor, avoid) {
      this.anchor = anchor;
      this.avoid = (avoid || []).filter(Boolean);
      if (!this.host.isConnected) (document.body || document.documentElement).appendChild(this.host);
      addEventListener("scroll", this.reposition, true);
      addEventListener("resize", this.reposition);
      clearInterval(this.timer);
      this.timer = setInterval(this.reposition, 500);
      this.reposition();
      this.applySettings();
    }

    detach() {
      clearInterval(this.timer);
      removeEventListener("scroll", this.reposition, true);
      removeEventListener("resize", this.reposition);
      this.host.remove();
    }

    applySettings() {
      this.host.style.display = settings.enabled && settings.pageBadge && !this.dismissed ? "" : "none";
    }

    // Put the pill next to the captcha without covering it, its input or the focused field.
    // Tries right/left/above/below the captcha and the input; in a cramped frame the pill
    // shrinks to its icon; as a last resort the spot with the least overlap wins.
    // After the user dragged it, the pill keeps that offset from the captcha.
    reposition() {
      if (!this.anchor || !this.anchor.isConnected) return;
      const ar = this.anchor.getBoundingClientRect();
      if (this.manual) {
        const l = Math.max(0, Math.min(innerWidth - this.pill.offsetWidth, ar.left + this.manual.dx));
        const top = Math.max(0, Math.min(innerHeight - this.pill.offsetHeight, ar.top + this.manual.dy));
        this.pill.style.left = `${Math.round(l)}px`;
        this.pill.style.top = `${Math.round(top)}px`;
        return;
      }
      // covering an input the user types into is 10x worse than covering the picture
      const keep = [[this.anchor, 1], ...(this.avoid || []).map((el) => [el, 10]), [document.activeElement, 10]]
        .filter(([el]) => el && el !== document.body && el !== document.documentElement && el.isConnected &&
          el !== this.host)
        .map(([el, weight]) => Object.assign(el.getBoundingClientRect().toJSON(), { weight }))
        .filter((r) => r.width > 0 && r.height > 0);
      const place = () => {
        const w = this.pill.offsetWidth || 180;
        const h = this.pill.offsetHeight || 30;
        const gap = 6;
        const vw = innerWidth;
        const vh = innerHeight;
        const cands = [];
        for (const r of keep.slice(0, 2)) {
          const midY = r.top + Math.min(r.height, 120) / 2 - h / 2;
          cands.push([r.right + gap, midY], [r.left - gap - w, midY],
            [r.right - w, r.top - h - gap], [r.left, r.top - h - gap],
            [r.left, r.bottom + gap], [r.right - w, r.bottom + gap]);
        }
        cands.push([vw - w - 4, 4], [vw - w - 4, vh - h - 4], [4, 4], [4, vh - h - 4]);
        let best = null;
        for (const [l0, t0] of cands) {
          const l = Math.max(4, Math.min(vw - w - 4, l0));
          const top = Math.max(4, Math.min(vh - h - 4, t0));
          let overlap = 0;
          for (const r of keep) {
            const ox = Math.max(0, Math.min(l + w, r.right) - Math.max(l, r.left));
            const oy = Math.max(0, Math.min(top + h, r.bottom) - Math.max(top, r.top));
            overlap += ox * oy * r.weight;
          }
          if (!best || overlap < best.overlap) best = { l, top, overlap };
          if (overlap === 0) break;
        }
        return best;
      };
      this.pill.classList.remove("compact");
      let spot = place();
      if (spot.overlap > 0) {
        this.pill.classList.add("compact"); // icon (+ buttons) only; the text stays in the tooltip
        spot = place();
      }
      this.pill.style.top = `${Math.round(spot.top)}px`;
      this.pill.style.left = `${Math.round(spot.l)}px`;
    }

    set(state, text, showButton) {
      const keepCls = ["compact", "dragging"].filter((c) => this.pill.classList.contains(c)).join(" ");
      this.pill.className = `pill ${state}${keepCls ? " " + keepCls : ""}`;
      this.pill.title = `Captchosa: ${text}`;
      this.msg.textContent = text;
      this.msg.title = `Captchosa: ${text}`;
      this.button.hidden = !showButton;
      this.reposition();
    }

    // Hide while hit-testing, so the pill can never be mistaken for a cover.
    async hidden(fn) {
      const prev = this.host.style.visibility;
      this.host.style.visibility = "hidden";
      try { return await fn(); } finally { this.host.style.visibility = prev; }
    }
  }

  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(String(fr.result).split(",", 2)[1] || "");
      fr.onerror = () => reject(fr.error);
      fr.readAsDataURL(blob);
    });
  }

  // Pixels of an already loaded <img>, without downloading it again.
  async function imageToBase64(img, timeoutMs) {
    if (!img.complete || !img.naturalWidth) {
      await new Promise((resolve, reject) => {
        const t = setTimeout(() => reject(Object.assign(new Error("image did not load"), { code: "IMAGE_NOT_LOADED" })),
          timeoutMs || 10000);
        img.addEventListener("load", () => { clearTimeout(t); resolve(); }, { once: true });
        img.addEventListener("error", () => { clearTimeout(t); reject(Object.assign(new Error("image failed to load"), { code: "IMAGE_NOT_LOADED" })); }, { once: true });
      });
    }
    const src = img.currentSrc || img.src || "";
    const m = /^data:image\/[a-z+.-]+;base64,(.*)$/i.exec(src);
    if (m) return m[1];
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext("2d").drawImage(img, 0, 0);
    try {
      return canvas.toDataURL("image/png").split(",", 2)[1];
    } catch (_) {
      throw Object.assign(new Error("the image is cross-origin and cannot be read"), { code: "IMAGE_TAINTED" });
    }
  }

  globalThis.CS = {
    get settings() { return settings; },
    loadSettings, onSettings, sleep, waitFor, solve, outcome, capture,
    badgeState: (state, title) => send({ type: "badge", state, title }, 5000),
    deepQueryAll, isVisible, hitTest, humanClick, setInputValue, Badge, imageToBase64, blobToBase64,
  };
})();
