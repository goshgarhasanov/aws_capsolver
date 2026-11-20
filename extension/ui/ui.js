// Popup and options page logic (both pages load this file; missing elements are skipped).
"use strict";

const { KINDS, normalize, originPattern } = globalThis.CAPTCHOSA;
const $ = (id) => document.getElementById(id);
const isOptions = document.body.classList.contains("options");

let settings = normalize(null);
let stats = {};
let log = [];
let dict = null;            // messages of an explicitly chosen language, else chrome.i18n
let grantedOrigins = new Set();
let lastTest = null;

// ------------------------------------------------------------ i18n

async function loadDict(lang) {
  if (lang === "auto") return null;
  try {
    const res = await fetch(chrome.runtime.getURL(`_locales/${lang}/messages.json`));
    return await res.json();
  } catch (_) {
    return null;
  }
}

function t(key) {
  if (dict && dict[key]) return dict[key].message;
  return chrome.i18n.getMessage(key) || key;
}

function uiLocale() {
  return settings.lang === "auto" ? chrome.i18n.getUILanguage() : settings.lang;
}

function applyI18n() {
  document.documentElement.lang = uiLocale();
  for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const el of document.querySelectorAll("[data-i18n-aria]")) el.setAttribute("aria-label", t(el.dataset.i18nAria));
  for (const el of document.querySelectorAll("[data-i18n-title]")) el.title = t(el.dataset.i18nTitle);
}

function ago(ts) {
  if (!ts) return t("never");
  const s = Math.round((ts - Date.now()) / 1000);
  const rtf = new Intl.RelativeTimeFormat(uiLocale(), { numeric: "auto", style: "short" });
  const abs = Math.abs(s);
  if (abs < 60) return rtf.format(s, "second");
  if (abs < 3600) return rtf.format(Math.round(s / 60), "minute");
  if (abs < 86400) return rtf.format(Math.round(s / 3600), "hour");
  return rtf.format(Math.round(s / 86400), "day");
}

const num = (n) => new Intl.NumberFormat(uiLocale()).format(n);

// ------------------------------------------------------------ storage

async function persist(patch) {
  settings = normalize(Object.assign({}, settings, patch));
  await chrome.storage.local.set({ settings });
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function makeSwitch(id, checked, label, onChange) {
  const wrap = el("label", "switch");
  const input = el("input");
  input.type = "checkbox";
  input.id = id;
  input.setAttribute("role", "switch");
  input.setAttribute("aria-label", label);
  input.checked = checked;
  input.addEventListener("change", () => onChange(input.checked));
  wrap.append(input, el("span", "track"));
  return wrap;
}

// ------------------------------------------------------------ render

// One small icon per captcha type (constant inline SVG, 20x20, currentColor).
const KIND_ICONS = {
  // AWS WAF picture grid: 3x3 tiles, two of them picked
  awsGrid: '<rect x="2" y="2" width="4.5" height="4.5" rx="1"/><rect x="7.75" y="2" width="4.5" height="4.5" rx="1" class="pick"/>' +
    '<rect x="13.5" y="2" width="4.5" height="4.5" rx="1"/><rect x="2" y="7.75" width="4.5" height="4.5" rx="1"/>' +
    '<rect x="7.75" y="7.75" width="4.5" height="4.5" rx="1"/><rect x="13.5" y="7.75" width="4.5" height="4.5" rx="1" class="pick"/>' +
    '<rect x="2" y="13.5" width="4.5" height="4.5" rx="1"/><rect x="7.75" y="13.5" width="4.5" height="4.5" rx="1"/>' +
    '<rect x="13.5" y="13.5" width="4.5" height="4.5" rx="1"/>',
  // AWS WAF token: a key
  awsToken: '<circle cx="6.5" cy="10" r="3.6" fill="none" stroke="currentColor" stroke-width="2"/>' +
    '<path d="M10 10h8M15 10v3.2M17.6 10v2.4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
  // BLS number grid: tiles with a number
  blsGrid: '<rect x="2" y="2" width="7" height="7" rx="1.5"/><rect x="11" y="2" width="7" height="7" rx="1.5" class="pick"/>' +
    '<rect x="2" y="11" width="7" height="7" rx="1.5" class="pick"/><rect x="11" y="11" width="7" height="7" rx="1.5"/>' +
    '<text x="14.5" y="7.6" class="glyph">7</text><text x="5.5" y="16.6" class="glyph">7</text>',
  // BLS Poland: digits on a wavy line
  blsText: '<path d="M1.5 14c3-4 5 3 8.5-1s5.5 2 8.5-1" fill="none" stroke="currentColor" stroke-width="1.4" opacity=".55"/>' +
    '<text x="10" y="11.5" class="glyph wide">123</text>',
  // general text captcha: "Aa" on a picture frame
  text: '<rect x="1.5" y="3.5" width="17" height="13" rx="2.5" fill="none" stroke="currentColor" stroke-width="1.6"/>' +
    '<text x="10" y="13.6" class="glyph wide">Aa</text>',
};

function kindIcon(kind) {
  const span = el("span", `kind-icon kind-${kind}`);
  span.setAttribute("aria-hidden", "true");
  // constant markup from KIND_ICONS only (no page or user data)
  span.innerHTML = `<svg viewBox="0 0 20 20" width="20" height="20" fill="currentColor">${KIND_ICONS[kind] || ""}</svg>`;
  return span;
}

function renderSolvers() {
  const box = $("solvers");
  if (!box) return;
  const focused = document.activeElement && box.contains(document.activeElement) ? document.activeElement.id : null;
  box.textContent = "";
  for (const kind of KINDS) {
    const st = stats[kind] || { solved: 0, failed: 0 };
    const on = settings.solvers[kind];
    const row = el("div", `solver${on ? "" : " off"}`);
    row.dataset.kind = kind;
    const info = el("div", "info");
    info.append(el("span", "name", t(`kind_${kind}`)));
    const last = st.last;
    const sub = el("span", `sub${last ? (last.ok ? " ok" : " err") : ""}`);
    if (last) {
      sub.append(el("span", "lastdot"), document.createTextNode(ago(last.ts)));
      sub.title = `${new Date(last.ts).toLocaleString(uiLocale())} – ${last.detail || ""}`;
    } else {
      sub.textContent = t(`desc_${kind}`);
    }
    if (isOptions && last) sub.append(document.createTextNode(` · ${t(`desc_${kind}`)}`));
    info.title = t(`desc_${kind}`);
    info.append(sub);
    const counts = el("div", "counts");
    const ok = el("span", `chip ok${st.solved ? "" : " zero"}`, `✓ ${num(st.solved)}`);
    ok.title = `${num(st.solved)} ${t("solvedCount")}`;
    const bad = el("span", `chip err${st.failed ? "" : " zero"}`, `✕ ${num(st.failed)}`);
    bad.title = `${num(st.failed)} ${t("failedCount")}`;
    counts.append(ok, bad);
    row.append(kindIcon(kind), info, counts, makeSwitch(`solver-${kind}`, on, t(`kind_${kind}`),
      (v) => persist({ solvers: Object.assign({}, settings.solvers, { [kind]: v }) })));
    box.append(row);
  }
  if (focused && $(focused)) $(focused).focus(); // keep keyboard focus across re-renders
}

function renderLog() {
  const ol = $("log");
  if (!ol) return;
  ol.textContent = "";
  $("logCount").textContent = String(log.length);
  if (!log.length) {
    ol.append(el("li", "empty", t("noActivity")));
    return;
  }
  for (const e of log.slice(0, 30)) {
    const li = el("li");
    li.append(el("span", `mark ${e.ok ? "ok" : "err"}`, e.ok ? "✓" : "✕"));
    const what = el("span", "what");
    what.append(el("b", "", t(`kind_${e.kind}`)), document.createTextNode(" "),
      el("span", "", `${e.host ? `${e.host} · ` : ""}${e.detail || ""}`));
    what.title = `${e.host || ""} ${e.detail || ""}`.trim();
    const when = el("span", "t", `${ago(e.ts)}${e.ms ? ` · ${(e.ms / 1000).toFixed(1)} s` : ""}`);
    li.append(what, when);
    ol.append(li);
  }
}

function setChecked(id, v) {
  const e = $(id);
  if (e) e.checked = v;
}

function renderForm() {
  setChecked("enabled", settings.enabled);
  document.body.classList.toggle("disabled", !settings.enabled);
  setChecked("autoSolve", settings.autoSolve);
  setChecked("submitAfterFill", settings.submitAfterFill);
  setChecked("blsGridSubmit", settings.blsGridSubmit);
  setChecked("pageBadge", settings.pageBadge);
  if ($("delayMs") && document.activeElement !== $("delayMs")) $("delayMs").value = Math.min(5000, settings.delayMs);
  if ($("delayOut")) $("delayOut").textContent = `${num(settings.delayMs)} ${t("ms")}`;
  paintRange();
  if ($("retries") && document.activeElement !== $("retries")) $("retries").value = settings.retries;
  if ($("lang")) $("lang").value = settings.lang;
  if ($("serviceUrl") && document.activeElement !== $("serviceUrl")) $("serviceUrl").value = settings.serviceUrl;
  if ($("clientKey") && document.activeElement !== $("clientKey")) $("clientKey").value = settings.clientKey;
  const v = chrome.runtime.getManifest();
  $("versionName").textContent = `v${v.version_name || v.version}`;
}

function paintRange() {
  const r = $("delayMs");
  if (r) r.style.setProperty("--fill", `${(Number(r.value) / Number(r.max)) * 100}%`);
}

function renderStatus() {
  const pill = $("statusPill");
  const link = $("setupLink");
  if (link) link.hidden = true;
  $("balanceBox").hidden = false;
  if (!settings.clientKey) {
    pill.className = "pill warn";
    $("connText").textContent = t("notConfigured");
    if (link) {
      $("balanceBox").hidden = true;
      link.hidden = false;
      link.title = t("noKey");
    }
    $("balance").textContent = "—";
    return;
  }
  if (!lastTest) {
    pill.className = "pill";
    $("connText").textContent = t("checking");
    return;
  }
  const h = lastTest.health || {};
  const b = lastTest.balance || {};
  const ok = !h.error && h.ok !== false;
  pill.className = `pill ${ok && !b.error ? "ok" : "err"}`;
  $("connText").textContent = ok ? t("connected") : t("offline");
  pill.title = ok ? `${settings.serviceUrl} · ${t("version")} ${h.version || "?"}` : `${h.error}: ${h.message}`;
  if (!b.error) {
    $("balance").textContent = b.balance === null || b.balance === undefined ? t("unlimited") : num(b.balance);
  } else {
    $("balance").textContent = "—";
    if (ok) pill.title = `${b.error}: ${b.message}`;
  }
}

function renderAll() {
  applyI18n();
  renderForm();
  renderStatus();
  renderSolvers();
  renderLog();
}

function showMessage(text, kind) {
  const m = $("message");
  if (!m) return;
  m.textContent = text || "";
  m.className = `message ${kind || ""}`;
}

// ------------------------------------------------------------ actions

async function testConnection(verbose) {
  if (verbose) showMessage(t("testing"), "");
  try {
    lastTest = await chrome.runtime.sendMessage({ type: "test" });
  } catch (err) {
    lastTest = { health: { error: "EXTENSION_ERROR", message: err.message }, balance: { error: "EXTENSION_ERROR" } };
  }
  renderStatus();
  if (!verbose) return;
  const h = lastTest.health || {};
  const b = lastTest.balance || {};
  if (!h.error && !b.error) showMessage(`${t("connected")} · ${t("balance")} ${$("balance").textContent}`, "ok");
  else if (!h.error) showMessage(`${b.error}: ${b.message}`, "err");
  else showMessage(`${h.error}: ${h.message}`, "err");
}

function saveConnection() {
  const url = $("serviceUrl").value.trim().replace(/\/+$/, "");
  const key = $("clientKey").value.trim();
  const pattern = originPattern(url);
  if (!pattern) {
    showMessage(t("badUrl"), "err");
    return;
  }
  const done = async () => {
    await persist({ serviceUrl: url, clientKey: key });
    showMessage(t("saved"), "ok");
    lastTest = null;
    renderStatus();
    testConnection(false);
  };
  if (grantedOrigins.has(pattern)) {
    done();
    return;
  }
  // must run synchronously inside the click to count as a user gesture
  chrome.permissions.request({ origins: [pattern] }).then((granted) => {
    if (granted) {
      grantedOrigins.add(pattern);
      done();
    } else {
      showMessage(t("permDenied"), "err");
    }
  }, (err) => showMessage(`${t("permDenied")}: ${err.message}`, "err"));
}

function on(id, event, fn) {
  const e = $(id);
  if (e) e.addEventListener(event, fn);
}

function bind() {
  on("enabled", "change", (e) => persist({ enabled: e.target.checked }));
  for (const id of ["autoSolve", "submitAfterFill", "blsGridSubmit", "pageBadge"]) {
    on(id, "change", (e) => persist({ [id]: e.target.checked }));
  }
  on("delayMs", "input", (e) => {
    $("delayOut").textContent = `${num(Number(e.target.value))} ${t("ms")}`;
    paintRange();
  });
  on("delayMs", "change", (e) => persist({ delayMs: e.target.value }));
  on("retries", "change", (e) => persist({ retries: e.target.value }));
  on("retriesDown", "click", () => persist({ retries: settings.retries - 1 }));
  on("retriesUp", "click", () => persist({ retries: settings.retries + 1 }));
  on("lang", "change", (e) => persist({ lang: e.target.value }));
  on("toggleKey", "click", () => {
    const k = $("clientKey");
    k.type = k.type === "password" ? "text" : "password";
    $("toggleKey").dataset.i18n = k.type === "password" ? "show" : "hide";
    $("toggleKey").textContent = t($("toggleKey").dataset.i18n);
  });
  on("save", "click", saveConnection);
  on("test", "click", () => testConnection(true));
  on("clearLog", "click", (e) => {
    e.preventDefault(); // inside <summary>: do not toggle
    chrome.storage.local.set({ log: [] });
  });
  const ALL_SITES = { origins: ["<all_urls>"] };
  const markAllowed = (yes) => {
    const b = $("allowAll");
    if (!b) return;
    b.disabled = yes;
    b.textContent = t(yes ? "allowAllSitesDone" : "allowAllSites");
  };
  if ($("allowAll")) chrome.permissions.contains(ALL_SITES).then(markAllowed, () => markAllowed(false));
  on("allowAll", "click", () => {
    // must run synchronously inside the click (user gesture)
    chrome.permissions.request(ALL_SITES).then((granted) => {
      markAllowed(granted);
      if (!granted) showMessage(t("allowAllSitesHint"), "err");
    }, (err) => showMessage(String(err && err.message || err), "err"));
  });
  on("resetStats", "click", async () => {
    await chrome.storage.local.set({ stats: {} });
    showMessage(t("statsReset"), "ok");
  });
  for (const id of ["openOptions", "setupLink"]) {
    on(id, "click", (e) => {
      e.preventDefault();
      chrome.runtime.openOptionsPage();
    });
  }
  const activity = $("activity");
  if (activity && !isOptions) {
    activity.addEventListener("toggle", () => {
      try { localStorage.setItem("activityOpen", activity.open ? "1" : "0"); } catch (_) { /* optional */ }
    });
    try { activity.open = localStorage.getItem("activityOpen") === "1"; } catch (_) { /* optional */ }
  }

  chrome.permissions.onAdded.addListener((p) => (p.origins || []).forEach((o) => grantedOrigins.add(o)));
  chrome.permissions.onRemoved.addListener((p) => (p.origins || []).forEach((o) => grantedOrigins.delete(o)));

  chrome.storage.onChanged.addListener(async (changes, area) => {
    if (area !== "local") return;
    if (changes.settings) {
      const prev = settings;
      settings = normalize(changes.settings.newValue);
      if (prev.lang !== settings.lang) dict = await loadDict(settings.lang);
      if (prev.serviceUrl !== settings.serviceUrl || prev.clientKey !== settings.clientKey) {
        lastTest = null;
        testConnection(false);
      }
    }
    if (changes.stats) stats = changes.stats.newValue || {};
    if (changes.log) log = changes.log.newValue || [];
    renderAll();
  });
}

async function init() {
  const data = await chrome.storage.local.get(["settings", "stats", "log"]);
  settings = normalize(data.settings);
  stats = data.stats || {};
  log = data.log || [];
  dict = await loadDict(settings.lang);
  try {
    const all = await chrome.permissions.getAll();
    grantedOrigins = new Set(all.origins || []);
  } catch (_) { /* treated as not granted */ }
  bind();
  renderAll();
  if (settings.clientKey) testConnection(false);
  // keep "x min ago" fresh while the page is open
  setInterval(() => { renderSolvers(); renderLog(); }, 30000);
}

init();
