// Runs in the page's MAIN world at document_start in every frame.
// It only observes: the AWS WAF captcha widget's own /problem and /verify answers
// and the challenge page's gokuProps are copied to the isolated content script
// via window.postMessage. Requests and responses are passed through untouched.
(function () {
  "use strict";
  if (window.__captchosaHooked) return;
  Object.defineProperty(window, "__captchosaHooked", { value: true });

  const TAG = "captchosa:main";
  const post = (type, data) => {
    try {
      window.postMessage({ source: TAG, type, data }, "*");
    } catch (_) {
      // structured clone failure: ignore
    }
  };

  const AWS_URL = /\/(problem|verify)(\?|$)/;

  function inspect(url, json) {
    if (!json || typeof json !== "object") return;
    const problem = json.problem && typeof json.problem === "object" ? json.problem
      : (json.assets && json.problem_type ? json : null);
    const isVerify = /\/verify(\?|$)/.test(url);
    if (isVerify) {
      post("aws-verify", {
        success: json.success === true,
        voucher: typeof json.captcha_voucher === "string" && json.captcha_voucher.length > 0,
      });
    }
    if (problem && problem.assets) {
      const a = problem.assets;
      post("aws-problem", {
        problem_type: String(problem.problem_type || ""),
        assets: { images: a.images, target: a.target },
        target_text: problem.localized_assets && problem.localized_assets.target0,
        fromVerify: isVerify,
      });
    }
  }

  function parseAndInspect(url, text) {
    if (typeof text !== "string" || text.length < 2 || text.length > 32 * 1024 * 1024) return;
    const c = text.trimStart()[0];
    if (c !== "{") return;
    try {
      inspect(url, JSON.parse(text));
    } catch (_) {
      // not JSON
    }
  }

  // ---- fetch
  const origFetch = window.fetch;
  if (typeof origFetch === "function") {
    const wrapped = function (input, init) {
      const p = origFetch.apply(this, arguments);
      let url = "";
      try {
        url = typeof input === "string" ? input : (input && input.url) || String(input);
      } catch (_) { /* opaque input */ }
      if (AWS_URL.test(url)) {
        p.then((res) => {
          try {
            res.clone().text().then((t) => parseAndInspect(url, t), () => {});
          } catch (_) { /* body already used or opaque */ }
        }, () => {});
      }
      return p;
    };
    try {
      Object.defineProperty(wrapped, "name", { value: "fetch" });
      wrapped.toString = origFetch.toString.bind(origFetch);
    } catch (_) { /* cosmetic only */ }
    window.fetch = wrapped;
  }

  // ---- XMLHttpRequest
  const XHR = window.XMLHttpRequest;
  if (XHR && XHR.prototype) {
    const origOpen = XHR.prototype.open;
    XHR.prototype.open = function (method, url) {
      try {
        const u = String(url);
        if (AWS_URL.test(u)) {
          this.addEventListener("load", () => {
            try {
              const rt = this.responseType;
              if (rt === "" || rt === "text") parseAndInspect(u, this.responseText);
              else if (rt === "json" && this.response) inspect(u, this.response);
            } catch (_) { /* unreadable body */ }
          });
        }
      } catch (_) { /* never break the page */ }
      return origOpen.apply(this, arguments);
    };
  }

  // ---- closed shadow roots of the AWS widget: keep them reachable for the solver
  const origAttach = Element.prototype.attachShadow;
  if (typeof origAttach === "function") {
    Element.prototype.attachShadow = function (init) {
      try {
        if (init && init.mode === "closed" && /^awswaf/i.test(this.tagName)) {
          init = Object.assign({}, init, { mode: "open" });
        }
      } catch (_) { /* keep the original init */ }
      return origAttach.call(this, init);
    };
  }

  // ---- AWS WAF challenge interstitial: window.gokuProps + challenge.js
  function reportGoku() {
    const g = window.gokuProps;
    if (!g || typeof g !== "object") return;
    let challengeJs = "";
    for (const s of document.scripts) {
      if (/challenge\.js(\?|$)/.test(s.src || "")) { challengeJs = s.src; break; }
    }
    post("aws-goku", {
      key: typeof g.key === "string" ? g.key : "",
      iv: typeof g.iv === "string" ? g.iv : "",
      context: typeof g.context === "string" ? g.context : "",
      challengeJs,
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", reportGoku, { once: true });
  } else {
    reportGoku();
  }
  // the isolated script may load after DOMContentLoaded; let it ask again
  window.addEventListener("message", (e) => {
    if (e.source === window && e.data && e.data.source === "captchosa:isolated" && e.data.type === "goku?") {
      reportGoku();
    }
  });
})();
