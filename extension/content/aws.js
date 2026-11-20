// AWS WAF: the visual grid captcha (AwsWafClassification) and the challenge
// interstitial token (AntiAwsWafTaskProxyLess). Isolated world, every frame.
(function () {
  "use strict";
  const CS = globalThis.CS;
  if (!CS || globalThis.__captchosaAws) return;
  globalThis.__captchosaAws = true;

  const KIND = "awsGrid";
  const CONFIRM_TEXT = /confirm|verify|submit|onayla|təsdiq|подтверд|bestätig|confirmer|confirmar|conferma|zatwierd/i;
  const NOT_CONFIRM = /audio|sound|refresh|reload|new\s*(puzzle|challenge)|help|info|close|cancel/i;
  const TILE_SELECTOR = 'button, [role="button"], [role="checkbox"], [role="gridcell"], [role="option"], li, td, canvas, img';

  const state = {
    problem: null,       // last intercepted grid problem
    generation: 0,       // bumps on every new problem; stale solves stop
    round: 0,            // puzzles answered for this widget
    startedAt: 0,
    waitingVerify: false,
    gaveUp: false,
    badge: null,
    tokenBusy: false,
  };

  function asList(v) {
    if (Array.isArray(v)) return v;
    if (typeof v === "string") {
      const t = v.trim();
      if (t.startsWith("[")) {
        try { const p = JSON.parse(t); return Array.isArray(p) ? p : []; } catch (_) { return []; }
      }
      return t ? [t] : [];
    }
    return [];
  }

  // ------------------------------------------------------------ widget DOM

  function widgetRoots() {
    const hosts = CS.deepQueryAll("awswaf-captcha, #captcha-container, [id^='awswaf'], [class*='awswaf']")
      .filter((el) => CS.isVisible(el) || el.shadowRoot);
    return hosts.length ? hosts : [document];
  }

  function tilePriority(el) {
    const tag = el.tagName;
    if (tag === "BUTTON" || el.hasAttribute("role")) return 3;
    if (tag === "LI" || tag === "TD") return 2;
    return 1;
  }

  function rowMajor(a, b) {
    const ra = a.getBoundingClientRect();
    const rb = b.getBoundingClientRect();
    if (Math.abs(ra.top - rb.top) > Math.min(ra.height, rb.height) / 2) return ra.top - rb.top;
    return ra.left - rb.left;
  }

  // The N grid cells: a group of N visible, similar, roughly square siblings.
  function findTiles(count) {
    // The real widget: N <button class="amzn-grid-cell"> laid over one <canvas>.
    const cells = CS.deepQueryAll(".amzn-grid-cell").filter((el) => CS.isVisible(el));
    if (cells.length === count) return { tiles: cells, canvas: null };
    let best = null;
    for (const root of widgetRoots()) {
      const scope = root.shadowRoot || root;
      const groups = new Map();
      for (const el of CS.deepQueryAll(TILE_SELECTOR, scope)) {
        if (!CS.isVisible(el)) continue;
        const r = el.getBoundingClientRect();
        if (r.width < 24 || r.height < 24 || r.width / r.height > 1.6 || r.height / r.width > 1.6) continue;
        // group siblings of the same kind: a canvas next to the cells must not hide them
        let byTag = groups.get(el.parentNode);
        if (!byTag) groups.set(el.parentNode, (byTag = new Map()));
        if (!byTag.has(el.tagName)) byTag.set(el.tagName, []);
        byTag.get(el.tagName).push(el);
      }
      for (const same of [...groups.values()].flatMap((m) => [...m.values()])) {
        if (same.length !== count) continue;
        const widths = same.map((e) => e.getBoundingClientRect().width);
        if (Math.max(...widths) / Math.min(...widths) > 1.3) continue;
        const score = tilePriority(same[0]);
        if (!best || score > best.score) best = { score, tiles: same.sort(rowMajor) };
      }
      if (best) break;
    }
    if (best) return { tiles: best.tiles, canvas: null };
    // Fallback: one square canvas painted with the whole grid
    for (const root of widgetRoots()) {
      for (const c of CS.deepQueryAll("canvas", root.shadowRoot || root)) {
        const r = c.getBoundingClientRect();
        if (CS.isVisible(c) && r.width >= 150 && Math.abs(r.width - r.height) < r.width * 0.15) {
          return { tiles: null, canvas: c };
        }
      }
    }
    return null;
  }

  function findConfirm(tiles) {
    const own = CS.deepQueryAll("#amzn-captcha-verify-button").find((el) => CS.isVisible(el));
    if (own) return own;
    let best = null;
    const roots = widgetRoots();
    for (const root of roots) {
      for (const el of CS.deepQueryAll('button, [role="button"], input[type="submit"], input[type="button"]',
        root.shadowRoot || root)) {
        if (!CS.isVisible(el) || (tiles && tiles.some((t) => t === el || t.contains(el) || el.contains(t)))) continue;
        const label = `${el.id} ${el.className} ${el.getAttribute("aria-label") || ""} ${el.textContent || ""} ${el.value || ""}`;
        let score = 0;
        if (/verify/i.test(`${el.id} ${el.className}`)) score += 3;
        if (CONFIRM_TEXT.test(label)) score += 2;
        if (NOT_CONFIRM.test(`${el.id} ${el.getAttribute("aria-label") || ""} ${el.textContent || ""}`)) score -= 4;
        if (score > 0 && (!best || score > best.score)) best = { score, el };
      }
      if (best) break;
    }
    return best && best.el;
  }

  function ensureBadge(anchor) {
    if (!state.badge) state.badge = new CS.Badge(() => solveCurrent(true));
    state.badge.attach(anchor);
    return state.badge;
  }

  // ------------------------------------------------------------ grid solving

  function fail(detail, record) {
    if (state.badge) state.badge.set("error", detail, !CS.settings.autoSolve);
    if (record) CS.outcome(KIND, false, detail, Date.now() - state.startedAt);
  }

  async function onProblem(data) {
    if (!data || typeof data !== "object") return;
    await CS.loadSettings();
    state.generation += 1;
    if (data.fromVerify) {
      state.round += 1;
    } else {
      state.round = 1;
      state.gaveUp = false;
      state.startedAt = Date.now();
    }
    state.problem = data;
    state.waitingVerify = false;
    if (state.round > CS.settings.retries) {
      state.gaveUp = true;
      fail(`gave up after ${CS.settings.retries} round(s)`, true);
      return;
    }
    const gen = state.generation;
    const found = await CS.waitFor(() => findTiles(asList(data.assets.images).length || 9), 10000, 200);
    if (gen !== state.generation) return;
    const anchorEl = found ? (found.canvas || found.tiles[0].parentElement || found.tiles[0]) : null;
    if (anchorEl) ensureBadge(anchorEl);
    await maybeAuto(gen);
  }

  async function maybeAuto(gen) {
    const s = CS.settings;
    if (!state.problem || state.gaveUp || state.waitingVerify) return;
    if (!s.enabled || !s.solvers.awsGrid) {
      if (state.badge) state.badge.set("idle", s.enabled ? "AWS grid solver is off" : "Captchosa is off", false);
      return;
    }
    if (state.tokenBusy) return;
    if (!s.autoSolve) {
      if (state.badge) state.badge.set("idle", `round ${state.round}: ready`, true);
      return;
    }
    if (state.badge) state.badge.set("solving", `waiting ${s.delayMs} ms`, false);
    await CS.sleep(s.delayMs);
    if (gen !== state.generation) return;
    await solveCurrent(false);
  }

  let solving = false;

  async function solveCurrent(manual) {
    const problem = state.problem;
    if (!problem || solving || state.waitingVerify) return;
    if (manual && state.gaveUp) { state.gaveUp = false; state.round = 1; }
    const s = CS.settings;
    if (!s.enabled || !s.solvers.awsGrid) return;
    const gen = state.generation;
    const family = problem.problem_type.split("-")[0];
    if (family !== "gridcaptcha") {
      fail(`UNSUPPORTED: ${family || "unknown"} puzzles are not supported`, true);
      return;
    }
    const images = asList(problem.assets.images);
    const target = String(asList(problem.assets.target)[0] || "");
    if (!images.length || !target) {
      fail("BAD_PROBLEM: no images/target in the puzzle", true);
      return;
    }
    solving = true;
    try {
      const found = findTiles(images.length);
      if (!found) {
        fail("WIDGET_NOT_FOUND: grid cells not found in the page", true);
        return;
      }
      if (state.badge) state.badge.set("solving", `round ${state.round}: solving "${target}"`, false);
      const res = await CS.solve(KIND, { question: `aws:grid:${target}`, images });
      if (gen !== state.generation) return;
      if (!res.ok) {
        fail(`${res.code}: ${res.message}`, false);
        return;
      }
      const objects = (res.solution && res.solution.objects) || [];
      const idx = [...new Set(objects.map(Number))].filter((i) => Number.isInteger(i) && i >= 0 && i < images.length);
      if (!idx.length) {
        fail("BAD_SOLUTION: the service returned no tiles", true);
        return;
      }
      const fresh = findTiles(images.length) || found;
      for (const i of idx) {
        if (gen !== state.generation) return;
        if (fresh.tiles) {
          await CS.humanClick(fresh.tiles[i]);
        } else {
          const r = fresh.canvas.getBoundingClientRect();
          const side = Math.round(Math.sqrt(images.length)) || 3;
          const cw = r.width / side;
          const ch = r.height / side;
          await CS.humanClick(fresh.canvas, {
            x: r.left + (i % side + 0.3 + Math.random() * 0.4) * cw,
            y: r.top + (Math.floor(i / side) + 0.3 + Math.random() * 0.4) * ch,
          });
        }
        await CS.sleep(150 + Math.random() * 250);
      }
      const confirm = findConfirm(fresh.tiles);
      if (!confirm) {
        fail("WIDGET_NOT_FOUND: confirm button not found", true);
        return;
      }
      await CS.sleep(200 + Math.random() * 300);
      state.waitingVerify = true;
      if (state.badge) state.badge.set("solving", `round ${state.round}: picked ${idx.join(",")} - verifying`, false);
      await CS.humanClick(confirm);
      // if the widget never answers, allow another attempt
      setTimeout(() => {
        if (gen === state.generation && state.waitingVerify) {
          state.waitingVerify = false;
          fail("TIMEOUT: the widget did not answer the confirm", true);
        }
      }, 30000);
    } finally {
      solving = false;
    }
  }

  function onVerify(data) {
    if (!data) return;
    state.waitingVerify = false;
    if (data.success && data.voucher) {
      const ms = Date.now() - state.startedAt;
      if (state.badge) state.badge.set("ok", `solved in ${state.round} round(s)`, false);
      CS.outcome(KIND, true, `accepted after ${state.round} round(s)`, ms);
      state.problem = null;
    } else if (!data.success && state.badge) {
      state.badge.set("solving", `round ${state.round}: wrong answer, next puzzle`, false);
    }
  }

  // ------------------------------------------------------------ "Begin" screen

  // Before the first puzzle the real widget only shows a Begin button (same id as the
  // later Submit button) and no grid. Press it once per page when auto-solve is on.
  let beginPressed = false;

  function beginButton() {
    if (state.problem || CS.deepQueryAll(".amzn-grid-cell").length) return null;
    const btn = CS.deepQueryAll("#amzn-captcha-verify-button").find((el) => CS.isVisible(el) && !el.disabled);
    return btn || null;
  }

  async function pressBegin() {
    const s = CS.settings;
    if (beginPressed || !s.enabled || !s.solvers.awsGrid || !s.autoSolve) return;
    const btn = beginButton();
    if (!btn) return;
    beginPressed = true;
    await CS.sleep(s.delayMs);
    const again = beginButton();
    if (again) await CS.humanClick(again);
  }

  function hasCaptchaWidget() {
    return !!(document.getElementById("captcha-container") ||
      CS.deepQueryAll("#amzn-captcha-verify-button, .amzn-captcha-modal, awswaf-captcha").length ||
      [...document.scripts].some((sc) => /\/(captcha|jsapi)\.js(\?|$)/.test(sc.src || "")));
  }

  // ------------------------------------------------------------ challenge token

  const TOKEN_GUARD = "captchosa:awsTokenTries";

  async function onGoku(data) {
    if (!data || (!data.key && !data.context) || state.gokuSeen) return;
    state.gokuSeen = true; // reported twice (page load + our query): handle once
    await CS.loadSettings();
    const s = CS.settings;
    if (!s.enabled || !s.solvers.awsToken || !s.autoSolve) return;
    if (document.readyState === "loading") {
      await new Promise((r) => document.addEventListener("DOMContentLoaded", r, { once: true }));
    }
    if (hasCaptchaWidget()) return; // CAPTCHA action: the grid solver handles it in the page
    let tries = 0;
    try { tries = Number(sessionStorage.getItem(TOKEN_GUARD)) || 0; } catch (_) { /* storage blocked */ }
    if (tries >= s.retries) return; // never loop reloads
    try { sessionStorage.setItem(TOKEN_GUARD, String(tries + 1)); } catch (_) { /* storage blocked */ }
    state.tokenBusy = true;
    const badge = new CS.Badge(() => {});
    badge.attach(document.body || document.documentElement);
    badge.set("solving", "AWS WAF token: solving", false);
    const started = Date.now();
    try {
      await CS.sleep(s.delayMs);
      const res = await CS.solve("awsToken", {
        websiteURL: location.href, awsKey: data.key, awsIv: data.iv, awsContext: data.context,
        awsChallengeJS: data.challengeJs, userAgent: navigator.userAgent,
      });
      if (!res.ok) {
        badge.set("error", `${res.code}: ${res.message}`, false);
        return;
      }
      const cookie = res.solution && res.solution.cookie;
      if (!cookie) {
        badge.set("error", "VOUCHER_ONLY: the service returned no cookie", false);
        CS.outcome("awsToken", false, "VOUCHER_ONLY: no aws-waf-token cookie", Date.now() - started);
        return;
      }
      if (!/^[\w\-.:~+/=]+$/.test(cookie)) {
        badge.set("error", "BAD_SOLUTION: unexpected cookie value", false);
        CS.outcome("awsToken", false, "BAD_SOLUTION: unexpected cookie value", Date.now() - started);
        return;
      }
      const secure = location.protocol === "https:" ? "; Secure" : "";
      document.cookie = `aws-waf-token=${cookie}; path=/; max-age=86400; SameSite=Lax${secure}`;
      badge.set("ok", "AWS WAF token set - reloading", false);
      await CS.outcome("awsToken", true, "aws-waf-token set", Date.now() - started);
      setTimeout(() => location.reload(), 600);
    } finally {
      state.tokenBusy = false;
      if (state.problem) maybeAuto(state.generation);
    }
  }

  // ------------------------------------------------------------ wiring

  window.addEventListener("message", (e) => {
    if (e.source !== window || !e.data || e.data.source !== "captchosa:main") return;
    const { type, data } = e.data;
    if (type === "aws-problem") onProblem(data);
    else if (type === "aws-verify") onVerify(data);
    else if (type === "aws-goku") onGoku(data);
  });

  CS.onSettings((s) => {
    if (state.badge) state.badge.applySettings();
    if (state.problem && !solving) maybeAuto(state.generation);
    if (!state.problem) pressBegin();
    if (!s.enabled && state.badge) state.badge.set("idle", "Captchosa is off", false);
  });

  // the Begin screen appears after captcha.js renders; check for it a few times
  let beginChecks = 0;
  const beginTimer = setInterval(async () => {
    if (++beginChecks > 60 || beginPressed || state.problem) { clearInterval(beginTimer); return; }
    if (!beginButton()) return;
    await CS.loadSettings();
    pressBegin();
  }, 500);

  // gokuProps may have been reported before this script listened
  const askGoku = () => window.postMessage({ source: "captchosa:isolated", type: "goku?" }, "*");
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", askGoku, { once: true });
  else askGoku();
})();
