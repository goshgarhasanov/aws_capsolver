// GeeTest v3 and v4 (slide, icon / word / space click, nine-grid, gobang).
// Isolated world, every frame. The pictures are read from the page (canvas pixels, loaded
// <img> elements, CSS background images from the memory cache) and only re-requested or
// screenshotted when the page cannot hand them over. The answer is performed like a person:
// an eased, slightly wobbly drag for the slider and paced clicks for everything else.
(function () {
  "use strict";
  const CS = globalThis.CS;
  if (!CS || globalThis.__captchosaGeeTest) return;
  globalThis.__captchosaGeeTest = true;

  const KIND = "geetest";
  const VERDICT_MS = 8000;      // how long to wait for GeeTest's verdict after answering
  const IMAGE_TIMEOUT_MS = 10000;
  const FAMILIES = ["slide", "icon", "word", "space", "nine", "gobang"];

  const state = {
    busy: false,
    attempts: 0,              // rounds answered for the current widget
    gaveUp: false,
    current: null,            // { ch, sig }
    done: new Set(),          // signatures of puzzles that were accepted
    badge: null,
    opened: new WeakSet(),    // "click to verify" buttons already pressed
  };

  // ------------------------------------------------------------ class helpers

  // GeeTest v4 appends a hash to class names (geetest_bg_3a8f21): match both forms. The hash
  // must contain a digit, so geetest_btn_click is not taken for geetest_btn.
  const hashRe = new Map();
  function hasGt(el, name) {
    if (!el || !el.classList) return false;
    let re = hashRe.get(name);
    if (!re) hashRe.set(name, (re = new RegExp(`^geetest_${name}(_(?=[0-9a-z]*[0-9])[0-9a-z]{4,10})?$`)));
    for (const c of el.classList) if (re.test(c)) return true;
    return false;
  }

  // GeeTest renders into the light DOM (no shadow roots), so a plain query is enough.
  function gtAll(name, root) {
    const scope = root && root.querySelectorAll ? root : document;
    return [...scope.querySelectorAll(`[class*="geetest_${name}"]`)].filter((el) => hasGt(el, name));
  }

  function gtVisible(name, root) {
    return gtAll(name, root).filter(CS.isVisible);
  }

  // The first visible element among several class names (in order of preference).
  function gtFirst(names, root) {
    for (const n of names) {
      const el = gtVisible(n, root)[0];
      if (el) return el;
    }
    return null;
  }

  function closestGt(el, names) {
    for (let n = el; n && n !== document.documentElement; n = n.parentElement || (n.getRootNode && n.getRootNode().host)) {
      if (names.some((name) => hasGt(n, name))) return n;
    }
    return null;
  }

  function rowMajor(a, b) {
    const ra = a.getBoundingClientRect();
    const rb = b.getBoundingClientRect();
    if (Math.abs(ra.top - rb.top) > Math.min(ra.height, rb.height) / 2) return ra.top - rb.top;
    return ra.left - rb.left;
  }

  function bgUrl(el) {
    const m = /url\(\s*(['"]?)(.*?)\1\s*\)/.exec(getComputedStyle(el).backgroundImage || "");
    return m && m[2] ? m[2] : null;
  }

  // ------------------------------------------------------------ detection

  // The widget type GeeTest wrote into its class names (v4: geetest_slide, geetest_icon, ...).
  function typeToken(el) {
    for (let n = el; n && n !== document.body; n = n.parentElement) {
      for (const f of ["slide", "icon", "word", "space", "nine", "winlinze", "gobang"]) {
        if (hasGt(n, f)) return f === "winlinze" ? "gobang" : f;
      }
    }
    return null;
  }

  function v3Slide() {
    const bg = gtVisible("canvas_bg")[0];
    if (!bg) return null;
    const handle = gtVisible("slider_button")[0];
    if (!handle) return null;
    const root = closestGt(bg, ["panel_box", "widget", "holder"]) || bg.parentElement;
    return {
      version: 3, family: "slide", root, bgEl: bg, handle,
      pieceEl: gtAll("canvas_slice", root)[0] || null,
      fullEl: gtAll("canvas_fullbg", root)[0] || null,
      avoid: [handle],
    };
  }

  function v4Slide() {
    const handle = gtFirst(["btn", "slider_button"]);
    if (!handle) return null;
    const box = closestGt(handle, ["box", "box_wrap", "captcha"]) || document;
    const bg = gtVisible("bg", box)[0];
    if (!bg || !bgUrl(bg)) return null;
    const piece = gtAll("slice_bg", box).find((el) => bgUrl(el)) || gtAll("slice", box)[0] || null;
    return { version: 4, family: "slide", root: closestGt(handle, ["box"]) || bg.parentElement, bgEl: bg, handle,
      pieceEl: piece, fullEl: null, avoid: [handle] };
  }

  function hintText(box) {
    const tips = gtFirst(["ques_tips", "tip_content", "tips"], box);
    return tips ? (tips.textContent || "").replace(/\s+/g, " ").trim() : "";
  }

  // word hints: the characters after the instruction ("请依次点击：天 地" -> ["天", "地"])
  function wordHints(text) {
    const tail = text.split(/[:：]/).pop() || "";
    const chars = tail.match(/[㐀-鿿豈-﫿]/g);
    return chars && chars.length <= 8 ? chars : [];
  }

  function v3Click() {
    const img = gtVisible("item_img").find((el) => el.tagName === "IMG" &&
      (closestGt(el, ["big_item"]) || gtVisible("item_img").length === 1));
    if (!img) return null;
    const root = closestGt(img, ["panel_box", "widget"]) || img.parentElement;
    const text = hintText(root);
    const hints = wordHints(text);
    // word click when the characters are written in the tip; otherwise the hints are in
    // the picture's bottom strip and the model cuts them out itself (icon)
    const token = typeToken(img);
    const family = token === "space" ? token : (token === "word" || hints.length ? "word" : "icon");
    return { version: 3, family, root, imageEl: img, hints: family === "word" ? hints : [],
      hintEls: [], submit: gtFirst(["commit"], root), avoid: [gtFirst(["commit"], root)] };
  }

  function v4Grid() {
    const items = gtVisible("item").filter((el) => !closestGt(el.parentElement || el, ["item"]));
    if (items.length !== 9 && items.length !== 16 && items.length !== 25 && items.length !== 36) return null;
    const box = closestGt(items[0], ["box", "box_wrap", "captcha", "panel_box", "widget"]) || items[0].parentElement;
    const token = typeToken(items[0]);
    const family = token === "gobang" || (token !== "nine" && items.length !== 9) ? "gobang" : "nine";
    const cells = items.sort(rowMajor);
    const ch = { version: closestGt(items[0], ["panel_box"]) ? 3 : 4, family, root: box, cells,
      board: closestGt(items[0], ["table_box", "board", "window"]) || items[0].parentElement,
      submit: gtFirst(["submit", "commit"], box) };
    ch.avoid = [ch.submit];
    if (family === "nine") {
      const tips = gtFirst(["ques_tips", "ques_back", "tips"], box);
      const pic = tips && ([...tips.querySelectorAll("img, canvas")].find(CS.isVisible) ||
        (bgUrl(tips) ? tips : null) || [...tips.querySelectorAll("*")].find((el) => bgUrl(el)));
      ch.targetEl = pic || null;
      const text = tips ? (tips.textContent || "").replace(/\s+/g, " ").trim() : "";
      ch.targetText = (text.split(/[:：]/).pop() || "").trim().slice(0, 64);
      if (!ch.targetEl && !ch.targetText) return null;
    }
    return ch;
  }

  function v4Click() {
    const submit = gtFirst(["submit", "commit"]);
    const box = submit ? (closestGt(submit, ["box", "box_wrap", "captcha"]) || document) : null;
    const bg = box && gtVisible("bg", box).find((el) => bgUrl(el) || el.tagName === "IMG" || el.tagName === "CANVAS");
    if (!bg) return null;
    const tips = gtFirst(["ques_tips", "tips"], box);
    const hintEls = tips ? [...tips.querySelectorAll("img, canvas")].filter(CS.isVisible) : [];
    const token = typeToken(bg);
    let family = FAMILIES.includes(token) && token !== "slide" && token !== "nine" && token !== "gobang" ? token : null;
    if (!family) family = hintEls.length ? "icon" : "word";
    const text = hintText(box);
    let hints = [];
    if (family === "word") hints = wordHints(text);
    else if (family === "space" && text) hints = [text.slice(0, 64)];
    return { version: 4, family, root: closestGt(bg, ["box"]) || bg.parentElement, imageEl: bg, hints,
      hintEls: family === "icon" ? hintEls.slice(0, 8) : [], submit, avoid: [submit] };
  }

  function detect() {
    if (!document.querySelector('[class*="geetest_"]')) return null;
    return v3Slide() || v4Slide() || v4Grid() || v3Click() || v4Click();
  }

  // "Click to verify" buttons that open the puzzle (v3 radar, v4 button).
  function openButton() {
    const btn = gtFirst(["radar_tip", "radar_btn", "btn_click"]);
    if (!btn || state.opened.has(btn)) return null;
    if (gtVisible("radar_success").length || gtVisible("success").length) return null;
    return btn;
  }

  // ------------------------------------------------------------ reading pictures

  function sameOrigin(url) {
    try { return new URL(url, location.href).origin === location.origin; } catch (_) { return false; }
  }

  function loadImage(url, cors) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      if (cors) img.crossOrigin = "anonymous";
      const timer = setTimeout(() => reject(Object.assign(new Error("image did not load"), { code: "IMAGE_NOT_LOADED" })),
        IMAGE_TIMEOUT_MS);
      img.onload = () => { clearTimeout(timer); resolve(img); };
      img.onerror = () => { clearTimeout(timer); reject(Object.assign(new Error("image failed to load"), { code: "IMAGE_TAINTED" })); };
      img.src = url;
    });
  }

  function canvasOf(source, w, h) {
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    c.getContext("2d").drawImage(source, 0, 0, w, h);
    return c;
  }

  // canvas -> {b64, w, h, canvas}; throws IMAGE_TAINTED for cross-origin pixels
  function fromCanvas(c) {
    try {
      return { b64: c.toDataURL("image/png").split(",", 2)[1], w: c.width, h: c.height, canvas: c };
    } catch (_) {
      throw Object.assign(new Error("tainted canvas"), { code: "IMAGE_TAINTED" });
    }
  }

  async function captureEl(el) {
    if (window.top !== window) {
      throw Object.assign(new Error("a cross-origin GeeTest picture inside a frame cannot be read"), { code: "IMAGE_TAINTED" });
    }
    const r0 = el.getBoundingClientRect();
    if (r0.top < 0 || r0.left < 0 || r0.bottom > innerHeight || r0.right > innerWidth) {
      el.scrollIntoView({ block: "center", inline: "nearest" });
      await CS.sleep(250);
    }
    const r = el.getBoundingClientRect();
    const grab = () => CS.capture({ x: r.left, y: r.top, w: r.width, h: r.height, vw: innerWidth });
    const res = state.badge ? await state.badge.hidden(grab) : await grab();
    if (!res.ok) throw Object.assign(new Error(res.message), { code: res.code || "CAPTURE_FAILED" });
    const img = await loadImage(`data:image/png;base64,${res.body}`, false);
    return { b64: res.body, w: img.naturalWidth, h: img.naturalHeight, captured: true };
  }

  // Pixels of a canvas / <img> / CSS background picture in its own resolution.
  async function readPicture(el, allowCapture) {
    try {
      if (el.tagName === "CANVAS") return fromCanvas(el);
      if (el.tagName === "IMG") {
        const b64 = await CS.imageToBase64(el, IMAGE_TIMEOUT_MS);
        return { b64, w: el.naturalWidth, h: el.naturalHeight, source: el };
      }
      const url = bgUrl(el);
      if (!url) throw Object.assign(new Error("no picture on the element"), { code: "WIDGET_NOT_FOUND" });
      // data: and same-origin URLs come from memory; a cross-origin one needs CORS (GeeTest's CDN allows it)
      const local = url.startsWith("data:") || url.startsWith("blob:") || sameOrigin(url);
      const img = await loadImage(url, !local);
      const out = fromCanvas(canvasOf(img, img.naturalWidth, img.naturalHeight));
      out.source = img;
      return out;
    } catch (err) {
      if (!allowCapture || (err.code !== "IMAGE_TAINTED" && err.code !== "IMAGE_NOT_LOADED")) throw err;
    }
    return captureEl(el);
  }

  // A v3 slice canvas is as wide as the picture with the piece drawn at its left; cut the
  // transparent columns on the right so the model gets the piece image as GeeTest serves it
  // (left edge = the canvas' left edge, which is what the answer positions).
  function trimRight(pic) {
    try {
      const data = pic.canvas.getContext("2d").getImageData(0, 0, pic.w, pic.h).data;
      let right = -1;
      for (let x = pic.w - 1; x >= 0 && right < 0; x--) {
        for (let y = 0; y < pic.h; y++) {
          if (data[(y * pic.w + x) * 4 + 3] > 40) { right = x; break; }
        }
      }
      if (right < 0 || right >= pic.w - 1) return pic;
      const c = document.createElement("canvas");
      c.width = right + 1;
      c.height = pic.h;
      c.getContext("2d").drawImage(pic.canvas, 0, 0);
      return fromCanvas(c);
    } catch (_) {
      return pic;
    }
  }

  // nine-grid / gobang: one picture of the whole grid (a shared sprite, or the cells drawn together)
  async function readGrid(ch) {
    const pics = ch.cells.map((cell) => {
      if (bgUrl(cell)) return cell;
      return [cell, ...cell.querySelectorAll("*")].find((el) => el.tagName === "IMG" || el.tagName === "CANVAS" || bgUrl(el)) || null;
    });
    const side = Math.round(Math.sqrt(ch.cells.length));
    const urls = pics.map((p) => (p && p.tagName !== "CANVAS" ? (p.tagName === "IMG" ? p.currentSrc || p.src : bgUrl(p)) : null));
    if (ch.family === "nine" && urls[0] && urls.every((u) => u === urls[0])) {
      return readPicture(pics[0], true); // one sprite holds the whole grid
    }
    const read = [];
    for (const p of pics) read.push(p ? await readPicture(p, false) : null);
    const cw = Math.max(...read.map((r) => (r ? r.w : 0)));
    const chh = Math.max(...read.map((r) => (r ? r.h : 0)));
    if (!cw || !chh) return captureEl(ch.board);
    const out = document.createElement("canvas");
    out.width = cw * side;
    out.height = chh * side;
    const g = out.getContext("2d");
    const boardBg = bgUrl(ch.board);
    if (boardBg) {
      try { g.drawImage(await loadImage(boardBg, !sameOrigin(boardBg)), 0, 0, out.width, out.height); } catch (_) { /* no board picture */ }
    } else {
      g.fillStyle = "#ffffff";
      g.fillRect(0, 0, out.width, out.height);
    }
    for (let i = 0; i < read.length; i++) {
      if (!read[i]) continue;
      const src = read[i].canvas || read[i].source;
      g.drawImage(src, (i % side) * cw, Math.floor(i / side) * chh, cw, chh);
    }
    return fromCanvas(out);
  }

  // ------------------------------------------------------------ signatures

  function picKey(el) {
    if (!el) return "";
    if (el.tagName === "IMG") return el.currentSrc || el.src || "";
    if (el.tagName === "CANVAS") {
      try {
        const g = el.getContext("2d");
        const w = el.width;
        const h = el.height;
        let key = `${w}x${h}:`;
        for (const [fx, fy] of [[0.13, 0.21], [0.37, 0.52], [0.61, 0.33], [0.85, 0.77], [0.5, 0.9]]) {
          key += g.getImageData(Math.floor(w * fx), Math.floor(h * fy), 1, 1).data.join(".") + ",";
        }
        return key;
      } catch (_) {
        return "tainted";
      }
    }
    const u = bgUrl(el) || "";
    return u.length > 80 ? `${u.length}:${u.slice(0, 40)}${u.slice(-40)}` : u;
  }

  function signature(ch) {
    let key = `${ch.version}|${ch.family}|`;
    if (ch.family === "slide") key += picKey(ch.bgEl);
    else if (ch.cells) key += ch.cells.map((c) => picKey([c, ...c.querySelectorAll("*")].find((e) => e.tagName === "IMG" || bgUrl(e)) || c)).join(",");
    else key += picKey(ch.imageEl);
    return key;
  }

  // ------------------------------------------------------------ human-like input

  const rand = (a, b) => a + Math.random() * (b - a);

  function fire(target, type, x, y, buttons) {
    const base = { bubbles: true, cancelable: true, composed: true, view: window, clientX: x, clientY: y,
      screenX: x + (window.screenX || 0), screenY: y + (window.screenY || 0), button: 0, buttons };
    const ptr = Object.assign({ pointerId: 1, pointerType: "mouse", isPrimary: true,
      pressure: buttons ? 0.5 : 0, width: 1, height: 1 }, base);
    target.dispatchEvent(new PointerEvent(`pointer${type}`, ptr));
    target.dispatchEvent(new MouseEvent(`mouse${type}`, base));
  }

  function targetAt(x, y, fallback) {
    const el = document.elementFromPoint(x, y);
    return el && !(el.tagName === "CAPTCHOSA-BADGE") ? el : fallback;
  }

  // Minimum-jerk position profile: smooth start, smooth stop.
  const minJerk = (t) => t * t * t * (10 - 15 * t + 6 * t * t);

  // Move the held pointer from (x0) to (x1) over `ms`, with a slow vertical wobble.
  async function glide(drag, x1, ms) {
    const x0 = drag.x;
    const steps = Math.max(4, Math.round(ms / 16));
    const phase = rand(0, Math.PI * 2);
    const amp = rand(0.6, 1.8);
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      const x = x0 + (x1 - x0) * minJerk(t) + (i < steps ? rand(-0.35, 0.35) : 0);
      drag.y = drag.y0 + amp * Math.sin(phase + t * Math.PI * rand(0.9, 1.1)) + rand(-0.3, 0.3);
      drag.x = x;
      fire(targetAt(x, drag.y, drag.handle), "move", x, drag.y, 1);
      await CS.sleep(rand(11, 22));
    }
    drag.x = x1;
  }

  // Drag the slider until the piece element's left edge sits at `targetX` (CSS px).
  // Closed loop: the piece/handle ratio differs between GeeTest versions and layouts, so it
  // is measured on the way and the end of the drag is corrected (a small human overshoot too).
  async function dragSlider(ch, targetX) {
    const handle = ch.handle;
    const hr = handle.getBoundingClientRect();
    const pieceLeft = () => (ch.pieceEl ? ch.pieceEl.getBoundingClientRect().left : null);
    const p0 = pieceLeft();
    const need = targetX - (p0 === null ? ch.bgEl.getBoundingClientRect().left : p0);
    let ratio = 1;
    if (p0 !== null) {
      // first guess from the layout: a small piece travels the free width of the picture
      const pr = ch.pieceEl.getBoundingClientRect();
      const br = ch.bgEl.getBoundingClientRect();
      const track = handle.parentElement ? handle.parentElement.getBoundingClientRect() : br;
      if (pr.width < br.width * 0.6 && track.width > hr.width * 2) {
        ratio = (br.width - pr.width) / Math.max(1, track.width - hr.width);
      }
    }
    const drag = { handle, x: hr.left + hr.width * rand(0.35, 0.65), y0: hr.top + hr.height * rand(0.35, 0.65) };
    drag.y = drag.y0;
    const startX = drag.x;
    fire(handle, "over", drag.x, drag.y, 0);
    fire(handle, "move", drag.x, drag.y, 0);
    await CS.sleep(rand(60, 140));
    fire(handle, "down", drag.x, drag.y, 1);
    await CS.sleep(rand(90, 200));
    const total = Math.min(2200, 650 + Math.abs(need) * rand(2.2, 3.2));
    // phase 1: most of the way, then measure how far the piece really moved
    await glide(drag, startX + (need / ratio) * rand(0.62, 0.72), total * 0.55);
    await CS.sleep(rand(18, 40));
    if (p0 !== null) {
      const moved = pieceLeft() - p0;
      const dx = drag.x - startX;
      if (Math.abs(dx) > 8 && Math.abs(moved) > 2) ratio = moved / dx;
    }
    // phase 2: a little past the target, then settle back
    const over = rand(2, 6) * Math.sign(need || 1);
    await glide(drag, startX + (need + over) / ratio, total * 0.35);
    await CS.sleep(rand(40, 110));
    await glide(drag, startX + need / ratio, rand(140, 260));
    // fine correction from what the page actually shows
    for (let i = 0; i < 4 && p0 !== null; i++) {
      await CS.sleep(rand(25, 50));
      const err = targetX - pieceLeft();
      if (Math.abs(err) <= 0.6) break;
      await glide(drag, drag.x + err / ratio, rand(70, 140));
    }
    await CS.sleep(rand(80, 220));
    fire(targetAt(drag.x, drag.y, handle), "up", drag.x, drag.y, 0);
    return p0 === null ? null : pieceLeft() - targetX;
  }

  // ------------------------------------------------------------ solving

  async function buildTask(ch) {
    const task = { question: `geetest:${ch.family}` };
    const extra = {};
    if (ch.family === "slide") {
      const bg = await readPicture(ch.bgEl, true);
      task.images = [bg.b64];
      extra.bg = bg;
      if (ch.pieceEl) {
        try {
          const piece = await readPicture(ch.pieceEl, false);
          task.piece = (piece.canvas && ch.pieceEl.tagName === "CANVAS" ? trimRight(piece) : piece).b64;
        } catch (_) {
          // the model can find the gap without the piece
        }
      }
      if (ch.fullEl) {
        try { task.fullBackground = (await readPicture(ch.fullEl, false)).b64; } catch (_) { /* optional */ }
      }
    } else if (ch.cells) {
      const grid = await readGrid(ch);
      task.images = [grid.b64];
      if (ch.family === "nine") {
        if (ch.targetEl) {
          try {
            task.targetImage = (await readPicture(ch.targetEl, true)).b64;
          } catch (_) {
            if (!ch.targetText) throw _;
          }
        }
        if (!task.targetImage) task.question = `geetest:nine:${ch.targetText}`;
      }
    } else {
      const pic = await readPicture(ch.imageEl, true);
      task.images = [pic.b64];
      extra.pic = pic;
      if (ch.hintEls.length) {
        const hints = [];
        for (const h of ch.hintEls) {
          try { hints.push((await readPicture(h, false)).b64); } catch (_) { hints.length = 0; break; }
        }
        if (hints.length) task.hints = hints;
      } else if (ch.hints.length) {
        task.hints = ch.hints;
      }
    }
    return { task, extra };
  }

  async function clickCell(cell) {
    await CS.humanClick(cell);
    await CS.sleep(rand(220, 520));
  }

  async function act(ch, solution, extra) {
    if (ch.family === "slide") {
      const x = Number(solution && solution.x);
      if (!Number.isFinite(x) || x < 0 || x > extra.bg.w) throw Object.assign(new Error("no x in the answer"), { code: "BAD_SOLUTION" });
      const br = ch.bgEl.getBoundingClientRect();
      const target = br.left + x * (br.width / extra.bg.w);
      // x: where the piece image's left edge must end (background pixels)
      const miss = await dragSlider(ch, target);
      return `slid to x=${Math.round(x)}${miss !== null && Math.abs(miss) > 1.5 ? ` (off by ${miss.toFixed(1)} px)` : ""}`;
    }
    if (ch.family === "nine" || ch.family === "gobang") {
      const side = Math.round(Math.sqrt(ch.cells.length));
      if (ch.family === "nine") {
        const idx = [...new Set(((solution && solution.objects) || []).map(Number))]
          .filter((i) => Number.isInteger(i) && i >= 0 && i < ch.cells.length);
        if (!idx.length) throw Object.assign(new Error("no cells in the answer"), { code: "BAD_SOLUTION" });
        for (const i of idx) await clickCell(ch.cells[i]);
        if (ch.submit && CS.isVisible(ch.submit)) {
          await CS.sleep(rand(200, 450));
          await CS.humanClick(ch.submit);
        }
        return `picked ${idx.join(",")}`;
      }
      const move = solution && solution.move;
      const ok = Array.isArray(move) && move.length === 2 && move.every((p) => Array.isArray(p) && p.length === 2 &&
        p.every((v) => Number.isInteger(v) && v >= 0 && v < side));
      if (!ok) throw Object.assign(new Error("no move in the answer"), { code: "BAD_SOLUTION" });
      const [[r1, c1], [r2, c2]] = move;
      await clickCell(ch.cells[r1 * side + c1]);
      await clickCell(ch.cells[r2 * side + c2]);
      if (ch.submit && CS.isVisible(ch.submit)) {
        await CS.sleep(rand(200, 450));
        await CS.humanClick(ch.submit);
      }
      return `moved ${r1},${c1} -> ${r2},${c2}`;
    }
    const pts = ((solution && solution.points) || []).filter((p) => Array.isArray(p) && p.length === 2 &&
      p.every((v) => Number.isFinite(Number(v)) && v >= 0));
    if (!pts.length || pts.length > 16) throw Object.assign(new Error("no points in the answer"), { code: "BAD_SOLUTION" });
    for (const [px, py] of pts) {
      const r = ch.imageEl.getBoundingClientRect();
      const x = r.left + Number(px) * (r.width / extra.pic.w);
      const y = r.top + Number(py) * (r.height / extra.pic.h);
      const hit = state.badge ? await state.badge.hidden(async () => targetAt(x, y, ch.imageEl)) : targetAt(x, y, ch.imageEl);
      const target = ch.root && ch.root.contains(hit) ? hit : ch.imageEl;
      await CS.humanClick(target, { x: x + rand(-1, 1), y: y + rand(-1, 1) });
      await CS.sleep(rand(320, 750));
    }
    if (ch.submit && CS.isVisible(ch.submit)) {
      await CS.sleep(rand(150, 400));
      await CS.humanClick(ch.submit);
    }
    return `clicked ${pts.length} point(s)`;
  }

  function succeeded(ch) {
    if (gtVisible("success").length || gtVisible("radar_success").length || gtVisible("success_radar_tip").length ||
        gtVisible("lock_success").length) return true;
    return !ch.root || !ch.root.isConnected || !CS.isVisible(ch.root);
  }

  // After answering: success, a fresh puzzle (wrong answer) or nothing at all.
  async function verdict(ch, sig) {
    const end = Date.now() + VERDICT_MS;
    while (Date.now() < end) {
      await CS.sleep(250);
      if (succeeded(ch)) return "ok";
      const now = detect();
      if (now && signature(now) !== sig) return "next";
    }
    return "none";
  }

  function badgeFor(ch) {
    if (!state.badge) state.badge = new CS.Badge(() => solve(true));
    state.badge.attach(ch.root || ch.bgEl || ch.imageEl, (ch.avoid || []).filter(Boolean));
    return state.badge;
  }

  function label(ch) {
    return `GeeTest v${ch.version} ${ch.family}`;
  }

  async function solve(manual) {
    if (state.busy) return;
    const s = CS.settings;
    if (!s.enabled || !s.solvers.geetest) return;
    const ch = detect();
    if (!ch) return;
    state.busy = true;
    const started = Date.now();
    const badge = badgeFor(ch);
    try {
      if (manual) { state.attempts = 0; state.gaveUp = false; }
      state.attempts += 1;
      const sig = signature(ch);
      state.current = { ch, sig };
      badge.set("solving", `${label(ch)}: reading the picture`, false);
      let built;
      try {
        built = await buildTask(ch);
      } catch (err) {
        const detail = `${err.code || "IMAGE_ERROR"}: ${err.message}`;
        badge.set("error", detail, true);
        CS.outcome(KIND, false, detail, Date.now() - started);
        state.gaveUp = true;
        return;
      }
      badge.set("solving", `${label(ch)}: solving (round ${state.attempts})`, false);
      const res = await CS.solve(KIND, built.task);
      if (!res.ok) {
        // no answer for this picture: ask GeeTest for another one while rounds are left
        const refresh = res.code === "ERROR_CAPTCHA_UNSOLVABLE" && state.attempts < s.retries
          ? gtFirst(["refresh", "refresh_1"]) : null;
        if (refresh) {
          badge.set("solving", `${label(ch)}: no answer, new picture`, false);
          await CS.sleep(rand(400, 900));
          await CS.humanClick(refresh);
          await CS.sleep(1200);
          return;
        }
        badge.set("error", `${res.code}: ${res.message}`, true);
        state.gaveUp = true;
        return;
      }
      // the puzzle may have been replaced while the service was thinking
      const fresh = detect();
      if (!fresh || signature(fresh) !== sig) {
        badge.set("solving", `${label(ch)}: the puzzle changed, reading again`, false);
        state.attempts -= 1;
        return;
      }
      let detail;
      try {
        detail = await act(ch, res.solution, built.extra);
      } catch (err) {
        detail = `${err.code || "ACTION_FAILED"}: ${err.message}`;
        badge.set("error", detail, true);
        CS.outcome(KIND, false, detail, Date.now() - started);
        state.gaveUp = true;
        return;
      }
      badge.set("solving", `${label(ch)}: ${detail} - verifying`, false);
      const v = await verdict(ch, sig);
      if (v === "ok") {
        state.done.add(sig);
        badge.set("ok", `${label(ch)}: solved in ${state.attempts} round(s)`, false);
        CS.outcome(KIND, true, `${label(ch)}: ${detail}, ${state.attempts} round(s)`, Date.now() - started);
        state.attempts = 0;
      } else if (v === "next") {
        if (state.attempts >= s.retries) {
          state.gaveUp = true;
          badge.set("error", `gave up after ${state.attempts} round(s); press Solve to try again`, true);
          CS.outcome(KIND, false, `${label(ch)}: wrong after ${state.attempts} round(s)`, Date.now() - started);
        } else {
          badge.set("solving", `${label(ch)}: wrong answer, next puzzle`, false);
        }
      } else {
        state.gaveUp = true;
        badge.set("error", `${label(ch)}: ${detail}, but GeeTest gave no verdict`, true);
        CS.outcome(KIND, false, `${label(ch)}: no verdict after ${detail}`, Date.now() - started);
      }
    } finally {
      state.busy = false;
      if (!state.gaveUp) schedule();
    }
  }

  let scanning = false;

  async function scan() {
    if (state.busy || scanning) return;
    scanning = true;
    try {
      await scanOnce();
    } finally {
      scanning = false;
    }
  }

  async function scanOnce() {
    await CS.loadSettings();
    const s = CS.settings;
    const ch = detect();
    if (!s.enabled || !s.solvers.geetest) {
      if (state.badge && ch) state.badge.set("idle", s.enabled ? "GeeTest solver is off" : "Captchosa is off", false);
      return;
    }
    if (!ch) {
      const btn = s.autoSolve ? openButton() : null;
      if (btn) {
        state.opened.add(btn);
        await CS.sleep(s.delayMs);
        if (!detect() && btn.isConnected && CS.isVisible(btn)) await CS.humanClick(btn);
      }
      return;
    }
    const sig = signature(ch);
    if (state.done.has(sig)) return;
    if (!state.current || state.current.sig !== sig) {
      // a different puzzle: after an accepted one (or a new widget) the round count restarts
      if (state.current && !state.current.ch.root.isConnected) { state.attempts = 0; state.gaveUp = false; }
    }
    const badge = badgeFor(ch);
    if (state.gaveUp) return; // the pill shows the error and the Solve button
    if (state.attempts >= s.retries) {
      state.gaveUp = true;
      badge.set("error", `gave up after ${state.attempts} round(s); press Solve to try again`, true);
      return;
    }
    if (!s.autoSolve) {
      badge.set("idle", `${label(ch)}: ready`, true);
      return;
    }
    badge.set("solving", `${label(ch)}: waiting ${s.delayMs} ms`, false);
    await CS.sleep(s.delayMs);
    const again = detect();
    if (!again || signature(again) !== sig || state.busy) return;
    await solve(false);
  }

  let timer = null;
  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(scan, 300);
  }

  function start() {
    new MutationObserver(schedule).observe(document.documentElement, {
      childList: true, subtree: true, attributes: true, attributeFilter: ["class", "style", "src"],
    });
    setInterval(() => { if (!state.busy) scan(); }, 2000);
    schedule();
  }

  CS.onSettings((s) => {
    if (state.badge) state.badge.applySettings();
    if (s.enabled && s.solvers.geetest && state.gaveUp && !state.busy) { state.gaveUp = false; state.attempts = 0; }
    schedule();
  });

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, { once: true });
  else start();
})();
