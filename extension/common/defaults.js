// Shared constants for the service worker, the popup and the content scripts.
// Classic script (no modules): it sets globalThis.CAPTCHOSA.
(function () {
  "use strict";

  // "text" covers every image text captcha, the BLS Poland login included (it used to be a
  // separate "blsText" solver; its switch is merged into "text" by the v3 migration)
  const KINDS = ["awsGrid", "awsToken", "blsGrid", "text", "geetest", "recaptcha"];
  const LANGS = ["auto", "en", "tr", "az", "ru"];

  // v2: the general text solver is on by default (it used to be off, so captchas on most
  // sites were never detected); stored v1 settings are migrated once.
  // v3: one text switch. v4: the reCAPTCHA v2 solver (on by default, like every new solver;
  // a missing switch takes its default, so v4 needs no data change)
  const SETTINGS_VERSION = 4;

  const DEFAULTS = {
    enabled: true,
    serviceUrl: "http://127.0.0.1:8790",
    clientKey: "",
    autoSolve: true,
    delayMs: 800,          // pause before solving a freshly detected captcha
    retries: 3,            // rounds per captcha (new puzzle after a wrong answer)
    solvers: { awsGrid: true, awsToken: true, blsGrid: true, text: true, geetest: true, recaptcha: true },
    blsGridSubmit: true,   // press Verify after picking BLS tiles (never with an empty password)
    submitAfterFill: false, // submit the form after filling a text captcha (auto-solve only)
    pageBadge: true,       // small status pill next to the captcha
    lang: "auto",          // "auto" = the browser language via chrome.i18n
  };

  function clampInt(value, lo, hi, fallback) {
    const n = Number.parseInt(value, 10);
    if (!Number.isFinite(n)) return fallback;
    return Math.min(hi, Math.max(lo, n));
  }

  // Merge stored (possibly partial or stale) settings over the defaults.
  function normalize(raw) {
    const s = Object.assign({}, DEFAULTS, raw && typeof raw === "object" ? raw : {});
    s.solvers = Object.assign({}, DEFAULTS.solvers, (raw && raw.solvers) || {});
    const v = raw && typeof raw === "object" ? Number(raw.v) || 1 : SETTINGS_VERSION;
    if (v < 2) s.solvers.text = true;                       // v2: general text on by default
    if (v < 3) {
      // v3: one text switch - on when either of the two old text solvers was on
      const rs = (raw && raw.solvers) || {};
      s.solvers.text = !(rs.blsText === false && (v >= 2 ? rs.text === false : rs.text !== true));
    }
    delete s.solvers.blsText;
    s.v = SETTINGS_VERSION;
    for (const k of KINDS) s.solvers[k] = DEFAULTS.solvers[k] ? s.solvers[k] !== false : s.solvers[k] === true;
    s.enabled = s.enabled !== false;
    s.autoSolve = s.autoSolve !== false;
    s.blsGridSubmit = s.blsGridSubmit !== false;
    s.submitAfterFill = s.submitAfterFill === true;
    delete s.blsTextSubmit;
    s.pageBadge = s.pageBadge !== false;
    s.delayMs = clampInt(s.delayMs, 0, 60000, DEFAULTS.delayMs);
    s.retries = clampInt(s.retries, 1, 10, DEFAULTS.retries);
    s.serviceUrl = String(s.serviceUrl || DEFAULTS.serviceUrl).trim().replace(/\/+$/, "");
    s.clientKey = String(s.clientKey || "").trim();
    s.lang = LANGS.includes(s.lang) ? s.lang : "auto";
    return s;
  }

  // Origin pattern for chrome.permissions, or null for an unusable URL.
  function originPattern(serviceUrl) {
    try {
      const u = new URL(serviceUrl);
      if (u.protocol !== "http:" && u.protocol !== "https:") return null;
      return `${u.protocol}//${u.hostname}/*`;
    } catch (_) {
      return null;
    }
  }

  globalThis.CAPTCHOSA = Object.freeze({ KINDS, LANGS, DEFAULTS, normalize, originPattern });
})();
