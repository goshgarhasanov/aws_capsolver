// Captchosa service worker: the only place that talks to the solving service.
// Nothing is ever sent anywhere except the user's configured service URL.
"use strict";

importScripts("common/defaults.js");

const { KINDS, normalize, originPattern } = globalThis.CAPTCHOSA;

const REQUEST_TIMEOUT_MS = 20000;   // one HTTP round trip
const CLASSIFY_TIMEOUT_MS = 70000;  // synchronous classification (server limit is 60 s)
const TOKEN_DEADLINE_MS = 150000;   // AntiAwsWafTask end to end (server limit is 120 s)
const POLL_INTERVAL_MS = 1500;
const LOG_SIZE = 40;

// Task type the service expects for every solver kind; content scripts never pick it.
const TASK_TYPES = {
  awsGrid: "AwsWafClassification",
  awsToken: "AntiAwsWafTaskProxyLess",
  blsGrid: "BlsClassification",
  text: "ImageToTextTask",
  geetest: "GeeTestClassification",
};
const GEETEST_QUESTION = /^geetest:(slide|icon|word|space|gobang|nine(:[^\x00-\x1f]{1,64})?)$/;
// ImageToTextTask module per text solver
// ImageToTextTask modules the text solver may ask for (BLS Poland login -> its own model)
const TEXT_MODULES = new Set(["common", "bls_poland"]);

class SolveError extends Error {
  constructor(code, message) {
    super(message || code);
    this.code = code;
  }
}

async function getSettings() {
  const { settings } = await chrome.storage.local.get("settings");
  return normalize(settings);
}

// ---------------------------------------------------------------- HTTP

async function fetchJson(url, init, timeoutMs) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(url, Object.assign({}, init, {
      signal: ctrl.signal, cache: "no-store", credentials: "omit", redirect: "error",
    }));
  } catch (err) {
    if (err && err.name === "AbortError") {
      throw new SolveError("TIMEOUT", `no answer from the service within ${Math.round(timeoutMs / 1000)} s`);
    }
    throw new SolveError("NETWORK_ERROR", `service unreachable (${(err && err.message) || err})`);
  } finally {
    clearTimeout(timer);
  }
  let body;
  try {
    body = await res.json();
  } catch (_) {
    throw new SolveError(res.ok ? "BAD_RESPONSE" : "HTTP_ERROR",
      `HTTP ${res.status}: the answer is not JSON`);
  }
  if (!body || typeof body !== "object") throw new SolveError("BAD_RESPONSE", "empty answer");
  if (body.errorId) {
    throw new SolveError(String(body.errorCode || "ERROR_UNKNOWN"),
      String(body.errorDescription || body.errorCode || "service error"));
  }
  if (!res.ok) throw new SolveError("HTTP_ERROR", `HTTP ${res.status}`);
  return body;
}

async function ensurePermission(settings) {
  const pattern = originPattern(settings.serviceUrl);
  if (!pattern) throw new SolveError("BAD_SERVICE_URL", "the service URL must be http(s)://host[:port]");
  const granted = await chrome.permissions.contains({ origins: [pattern] });
  if (!granted) {
    throw new SolveError("PERMISSION_MISSING",
      "access to the service URL is not granted; press Save in the options page");
  }
}

function api(settings, path, payload, timeoutMs) {
  return fetchJson(settings.serviceUrl + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(Object.assign({ clientKey: settings.clientKey }, payload)),
  }, timeoutMs || REQUEST_TIMEOUT_MS);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function runTask(settings, kind, task) {
  const sync = kind !== "awsToken";
  const deadline = Date.now() + (sync ? CLASSIFY_TIMEOUT_MS : TOKEN_DEADLINE_MS);
  const created = await api(settings, "/createTask", { task }, sync ? CLASSIFY_TIMEOUT_MS : REQUEST_TIMEOUT_MS);
  if (created.solution) return created.solution;
  if (!created.taskId) throw new SolveError("BAD_RESPONSE", "createTask returned neither solution nor taskId");
  while (Date.now() < deadline) {
    await sleep(POLL_INTERVAL_MS);
    const left = Math.max(1000, Math.min(REQUEST_TIMEOUT_MS, deadline - Date.now()));
    const res = await api(settings, "/getTaskResult", { taskId: created.taskId }, left);
    if (res.status === "ready") {
      if (!res.solution) throw new SolveError("BAD_RESPONSE", "ready without a solution");
      return res.solution;
    }
  }
  throw new SolveError("TIMEOUT", "the task did not finish in time");
}

// ---------------------------------------------------------------- task validation

function str(v, max) {
  return typeof v === "string" && v.length > 0 && v.length <= max;
}

// Rebuild the task from known fields only, so a page cannot smuggle anything else out.
function buildTask(kind, raw) {
  const t = raw && typeof raw === "object" ? raw : {};
  const b64list = (v, min, max) => Array.isArray(v) && v.length >= min && v.length <= max &&
    v.every((x) => str(x, 4 * 1024 * 1024));
  switch (kind) {
    case "awsGrid":
      if (!str(t.question, 200) || !b64list(t.images, 1, 30)) break;
      return { type: TASK_TYPES.awsGrid, question: t.question, images: t.images };
    case "blsGrid":
      if (!/^\d{3}$/.test(String(t.target || "")) || !b64list(t.images, 9, 9)) break;
      return { type: TASK_TYPES.blsGrid, question: `bls:grid:${t.target}`, images: t.images };
    case "text": {
      if (!str(t.body, 4 * 1024 * 1024)) break;
      const module = TEXT_MODULES.has(t.module) ? t.module : "common";
      // case: true = case-sensitive (the answer keeps upper/lower case exactly as read)
      return { type: TASK_TYPES.text, body: t.body, module, case: true };
    }
    case "geetest": {
      if (!str(t.question, 80) || !GEETEST_QUESTION.test(t.question) || !b64list(t.images, 1, 1)) break;
      const out = { type: TASK_TYPES.geetest, question: t.question, images: t.images };
      for (const k of ["piece", "fullBackground", "targetImage"]) {
        if (str(t[k], 4 * 1024 * 1024)) out[k] = t[k];
      }
      if (Number.isInteger(t.y) && t.y >= 0 && t.y <= 2000 && t.question === "geetest:slide") out.y = t.y;
      if (Number.isInteger(t.count) && t.count >= 1 && t.count <= 9 && t.question.startsWith("geetest:nine")) out.count = t.count;
      if (Array.isArray(t.hints) && t.hints.length) {
        if (t.hints.length > 8 || !t.hints.every((h) => str(h, 4 * 1024 * 1024))) break;
        out.hints = t.hints;
      }
      return out;
    }
    case "awsToken": {
      if (!str(t.websiteURL, 2048) || !/^https?:\/\//i.test(t.websiteURL)) break;
      const out = { type: TASK_TYPES.awsToken, websiteURL: t.websiteURL };
      for (const k of ["awsKey", "awsIv", "awsContext", "awsChallengeJS", "userAgent"]) {
        if (str(t[k], 8192)) out[k] = t[k];
      }
      return out;
    }
  }
  throw new SolveError("BAD_TASK", `the page data for ${kind} is incomplete`);
}

// ---------------------------------------------------------------- stats + log

let storageChain = Promise.resolve();

function record(kind, ok, detail, ms, host) {
  storageChain = storageChain.then(async () => {
    const { stats = {}, log = [] } = await chrome.storage.local.get(["stats", "log"]);
    const s = stats[kind] || { solved: 0, failed: 0 };
    if (ok) s.solved += 1; else s.failed += 1;
    s.last = { ok, detail, ms, ts: Date.now() };
    stats[kind] = s;
    log.unshift({ ts: Date.now(), kind, ok, detail, ms, host });
    await chrome.storage.local.set({ stats, log: log.slice(0, LOG_SIZE) });
  }).catch((err) => console.warn("captchosa: stats update failed", err));
  return storageChain;
}

// ---------------------------------------------------------------- toolbar badge

const BADGES = {
  on: { text: "", color: "#16a34a", icon: "icon" },
  off: { text: "OFF", color: "#6b7280", icon: "inactive" },
  solving: { text: "…", color: "#2563eb", icon: "solving" },
  ok: { text: "✓", color: "#16a34a", icon: "icon" },
  error: { text: "ERR", color: "#dc2626", icon: "error" },
};

function iconPaths(name) {
  const out = {};
  for (const size of [16, 32, 48]) out[size] = name === "icon" ? `icons/icon${size}.png` : `icons/${name}-${size}.png`;
  return out;
}
const okTimers = new Map();
const touchedTabs = new Set(); // tabs with a tab-specific badge

async function setBadge(tabId, state, title) {
  const b = BADGES[state] || BADGES.on;
  const opts = tabId == null ? {} : { tabId };
  try {
    await chrome.action.setBadgeText(Object.assign({ text: b.text }, opts));
    await chrome.action.setBadgeBackgroundColor(Object.assign({ color: b.color }, opts));
    await chrome.action.setTitle(Object.assign({ title: title ? `Captchosa: ${title}` : "Captchosa" }, opts));
    await chrome.action.setIcon(Object.assign({ path: iconPaths(b.icon) }, opts));
  } catch (_) {
    // the tab may be gone already
  }
}

async function tabBadge(tabId, state, title) {
  if (tabId == null || tabId < 0) return;
  touchedTabs.add(tabId);
  clearTimeout(okTimers.get(tabId));
  okTimers.delete(tabId);
  await setBadge(tabId, state, title);
  if (state === "ok" || state === "on") {
    okTimers.set(tabId, setTimeout(() => {
      okTimers.delete(tabId);
      refreshTabBadge(tabId);
    }, state === "ok" ? 5000 : 0));
  }
}

// back to the global look (on/off) for one tab
async function refreshTabBadge(tabId) {
  const s = await getSettings();
  await setBadge(tabId, s.enabled ? "on" : "off");
}

async function refreshGlobalBadge() {
  const s = await getSettings();
  await setBadge(null, s.enabled ? "on" : "off", s.enabled ? "enabled" : "disabled");
  for (const tabId of touchedTabs) await setBadge(tabId, s.enabled ? "on" : "off");
}

// ---------------------------------------------------------------- handlers

async function handleSolve(msg, sender) {
  const kind = msg.kind;
  if (!KINDS.includes(kind)) throw new SolveError("BAD_TASK", "unknown solver kind");
  const settings = await getSettings();
  if (!settings.enabled || !settings.solvers[kind]) throw new SolveError("DISABLED", `${kind} is switched off`);
  if (!settings.clientKey) throw new SolveError("NOT_CONFIGURED", "set the client key in the options page");
  const task = buildTask(kind, msg.task);
  await ensurePermission(settings);
  const tabId = sender.tab ? sender.tab.id : null;
  let host = "";
  try { host = new URL(sender.url || (sender.tab && sender.tab.url) || "").host; } catch (_) { /* opaque */ }
  await tabBadge(tabId, "solving", `solving ${kind}`);
  const started = Date.now();
  try {
    const solution = await runTask(settings, kind, task);
    const ms = Date.now() - started;
    return { ok: true, solution, ms, host };
  } catch (err) {
    const ms = Date.now() - started;
    await tabBadge(tabId, "error", `${err.code || "ERROR"}: ${err.message}`);
    await record(kind, false, `${err.code || "ERROR"}: ${err.message}`, ms, host);
    throw err;
  }
}

// The page reports the final outcome (e.g. AWS accepted the answer or asked again).
async function handleOutcome(msg, sender) {
  if (!KINDS.includes(msg.kind)) return {};
  const tabId = sender.tab ? sender.tab.id : null;
  let host = "";
  try { host = new URL(sender.url || "").host; } catch (_) { /* opaque */ }
  const detail = String(msg.detail || "").slice(0, 300);
  await tabBadge(tabId, msg.ok ? "ok" : "error", detail);
  await record(msg.kind, !!msg.ok, detail, Number(msg.ms) || 0, host);
  return {};
}

// Cut a rectangle out of a screenshot of the visible tab: used for captcha pictures the
// page cannot hand over (cross-origin images / tainted canvases). Top frame only.
async function handleCapture(msg, sender) {
  const settings = await getSettings();
  if (!settings.enabled) throw new SolveError("DISABLED", "Captchosa is off");
  if (!sender.tab || sender.frameId !== 0) {
    throw new SolveError("CAPTURE_FAILED", "only captchas in the top frame can be captured");
  }
  const r = msg.rect && typeof msg.rect === "object" ? msg.rect : {};
  const x = Number(r.x); const y = Number(r.y); const w = Number(r.w); const h = Number(r.h);
  const vw = Number(r.vw);
  if (![x, y, w, h, vw].every(Number.isFinite) || w < 8 || h < 8 || w > 1600 || h > 900 || vw < 100) {
    throw new SolveError("BAD_TASK", "bad capture rectangle");
  }
  const tab = await chrome.tabs.get(sender.tab.id);
  if (!tab.active) throw new SolveError("CAPTURE_FAILED", "the tab is not in front");
  let dataUrl;
  try {
    dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: "png" });
  } catch (err) {
    throw new SolveError("CAPTURE_DENIED", "this captcha picture comes from another site; open the " +
      "Captchosa options and press \"Allow on all sites\" so it can be read (" + ((err && err.message) || err) + ")");
  }
  const bmp = await createImageBitmap(await (await fetch(dataUrl)).blob());
  const scale = bmp.width / vw; // device pixels per CSS pixel (DPR and zoom)
  const sx = Math.max(0, Math.floor(x * scale));
  const sy = Math.max(0, Math.floor(y * scale));
  const sw = Math.min(bmp.width - sx, Math.ceil(w * scale));
  const sh = Math.min(bmp.height - sy, Math.ceil(h * scale));
  if (sw < 8 || sh < 8) throw new SolveError("CAPTURE_FAILED", "the captcha is not on screen");
  const canvas = new OffscreenCanvas(sw, sh);
  canvas.getContext("2d").drawImage(bmp, sx, sy, sw, sh, 0, 0, sw, sh);
  bmp.close();
  const bytes = new Uint8Array(await (await canvas.convertToBlob({ type: "image/png" })).arrayBuffer());
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return { body: btoa(bin) };
}

async function handleTest() {
  const settings = await getSettings();
  const out = { serviceUrl: settings.serviceUrl };
  try {
    await ensurePermission(settings);
  } catch (err) {
    out.health = { error: err.code, message: err.message };
    out.balance = { error: err.code, message: err.message };
    return out;
  }
  try {
    out.health = await fetchJson(settings.serviceUrl + "/health", { method: "GET" }, REQUEST_TIMEOUT_MS);
  } catch (err) {
    out.health = { error: err.code || "ERROR", message: err.message };
  }
  if (!settings.clientKey) {
    out.balance = { error: "NOT_CONFIGURED", message: "no client key set" };
  } else {
    try {
      const b = await api(settings, "/getBalance", {}, REQUEST_TIMEOUT_MS);
      out.balance = { balance: b.balance, usage: b.usage };
    } catch (err) {
      out.balance = { error: err.code || "ERROR", message: err.message };
    }
  }
  return out;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || typeof msg !== "object" || sender.id !== chrome.runtime.id) return false;
  let work;
  if (msg.type === "solve") work = handleSolve(msg, sender);
  else if (msg.type === "outcome") work = handleOutcome(msg, sender);
  else if (msg.type === "capture") work = handleCapture(msg, sender);
  else if (msg.type === "badge") work = tabBadge(sender.tab ? sender.tab.id : null, msg.state, msg.title);
  else if (msg.type === "test") work = handleTest();
  else return false;
  work.then(
    (value) => sendResponse(Object.assign({ ok: true }, value || {})),
    (err) => sendResponse({ ok: false, code: err.code || "ERROR", message: err.message || String(err) }),
  );
  return true; // async response
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.settings) refreshGlobalBadge();
});

chrome.tabs.onRemoved.addListener((tabId) => {
  clearTimeout(okTimers.get(tabId));
  okTimers.delete(tabId);
  touchedTabs.delete(tabId);
});

// a new page in the tab starts from the global look again
chrome.tabs.onUpdated.addListener((tabId, info) => {
  if (info.status === "loading" && touchedTabs.has(tabId)) {
    clearTimeout(okTimers.get(tabId));
    okTimers.delete(tabId);
    refreshTabBadge(tabId);
  }
});

chrome.runtime.onInstalled.addListener(async (details) => {
  const { settings } = await chrome.storage.local.get("settings");
  await chrome.storage.local.set({ settings: normalize(settings) });
  await refreshGlobalBadge();
  if (details.reason === "install") chrome.runtime.openOptionsPage();
});

chrome.runtime.onStartup.addListener(refreshGlobalBadge);
