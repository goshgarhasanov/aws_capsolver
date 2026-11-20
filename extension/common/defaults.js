// Shared constants for the service worker, the popup and the content scripts.
// Classic script (no modules): it sets globalThis.CAPTCHOSA.
(function () {
  "use strict";

  const KINDS = ["awsGrid", "awsToken", "blsGrid", "blsText", "text"];
  const LANGS = ["auto", "en", "tr", "az", "ru"];

  // v2: the general text solver is on by default (it used to be off, so captchas on most
  // sites were never detected); stored v1 settings are migrated once.
  const SETTINGS_VERSION = 2;

  const DEFAULTS = {
    enabled: true,
    serviceUrl: "http://127.0.0.1:8790",
    clientKey: "",
    autoSolve: true,
    delayMs: 800,          // pause before solving a freshly detected captcha
    retries: 3,            // rounds per captcha (new puzzle after a wrong answer)
    solvers: { awsGrid: true, awsToken: true, blsGrid: true, blsText: true, text: true },
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
    if (raw && typeof raw === "object" && !(Number(raw.v) >= SETTINGS_VERSION)) s.solvers.text = true;
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
