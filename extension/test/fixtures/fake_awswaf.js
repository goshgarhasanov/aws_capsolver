// A stand-in for the AWS WAF captcha widget used ONLY by the local test fixtures.
// <awswaf-captcha data-shadow="open|closed" data-transport="fetch|xhr" data-mode="wrongfirst">
(function () {
  "use strict";
  const sid = Math.random().toString(36).slice(2);

  function request(transport, method, url, body) {
    if (transport === "xhr") {
      return new Promise((resolve, reject) => {
        const x = new XMLHttpRequest();
        x.open(method, url);
        x.setRequestHeader("Content-Type", "text/plain;charset=UTF-8");
        x.onload = () => resolve(JSON.parse(x.responseText));
        x.onerror = reject;
        x.send(body || null);
      });
    }
    return fetch(url, { method, body, headers: { "Content-Type": "text/plain;charset=UTF-8" } }).then((r) => r.json());
  }

  class FakeCaptcha extends HTMLElement {
    connectedCallback() {
      this.transport = this.dataset.transport || "fetch";
      this.mode = this.dataset.mode || "";
      this.root = this.attachShadow({ mode: this.dataset.shadow || "open" });
      this.load();
    }

    async load() {
      const p = await request(this.transport, "GET", `/aws/problem?kind=visual&sid=${sid}&mode=${this.mode}`);
      this.render(p);
    }

    render(problem) {
      this.problem = problem;
      const images = JSON.parse(problem.assets.images);
      const target = problem.localized_assets.target0;
      this.root.innerHTML = `
        <style>
          .box { width: 340px; border: 1px solid #ccc; padding: 10px; font-family: sans-serif; }
          .grid { display: grid; grid-template-columns: repeat(3, 104px); gap: 4px; }
          .grid button { width: 104px; height: 104px; padding: 0; border: 3px solid transparent; background: #eee; }
          .grid button[aria-pressed=true] { border-color: #f90; }
          canvas { width: 98px; height: 98px; display: block; }
          .bar { display: flex; justify-content: space-between; margin-top: 8px; }
        </style>
        <div class="box">
          <p id="instructions">Choose all <b>${target}</b></p>
          <div class="grid" role="group"></div>
          <div class="bar">
            <button id="amzn-btn-audio-internal" aria-label="Switch to audio problem">Audio</button>
            <button id="amzn-btn-refresh-internal" aria-label="Get new problem">New</button>
            <button id="amzn-btn-verify-internal">Confirm</button>
          </div>
          <p id="status"></p>
        </div>`;
      const grid = this.root.querySelector(".grid");
      images.forEach((b64, i) => {
        const btn = document.createElement("button");
        btn.setAttribute("aria-pressed", "false");
        btn.setAttribute("aria-label", `Image ${i + 1}`);
        const c = document.createElement("canvas");
        c.width = 100; c.height = 100;
        const img = new Image();
        img.onload = () => c.getContext("2d").drawImage(img, 0, 0, 100, 100);
        img.src = "data:image/jpeg;base64," + b64;
        btn.appendChild(c);
        btn.addEventListener("click", () => {
          btn.setAttribute("aria-pressed", btn.getAttribute("aria-pressed") === "true" ? "false" : "true");
        });
        grid.appendChild(btn);
      });
      this.root.getElementById("amzn-btn-verify-internal").addEventListener("click", () => this.verify());
      this.root.getElementById("amzn-btn-refresh-internal").addEventListener("click", () => this.load());
    }

    async verify() {
      const picked = [...this.root.querySelectorAll(".grid button")]
        .map((b, i) => (b.getAttribute("aria-pressed") === "true" ? i : -1)).filter((i) => i >= 0);
      const body = JSON.stringify({ state: this.problem.state, key: this.problem.key,
        hmac_tag: this.problem.hmac_tag, client_solution: picked });
      const res = await request(this.transport, "POST", `/aws/verify?sid=${sid}`, body);
      if (res.success && res.captcha_voucher) {
        this.root.innerHTML = '<p id="done">Solved</p>';
        document.title = "AWS SOLVED";
      } else if (res.problem) {
        this.render(res.problem);
      }
    }
  }
  customElements.define("awswaf-captcha", FakeCaptcha);
})();
