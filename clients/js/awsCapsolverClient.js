/**
 * Minimal client for the aws_capsolver HTTP API (CapSolver-compatible).
 *
 * No dependencies: uses the global fetch (Node >= 18, Deno, Bun, browsers).
 * Works as an ES module:
 *
 *   import { AwsCapsolverClient, CapsolverError } from "./awsCapsolverClient.js";
 *   const client = new AwsCapsolverClient({ baseUrl: "http://127.0.0.1:8790", clientKey: "MY-KEY" });
 *   const { cookie, userAgent } = await client.solveAwsWaf("https://protected.example/", {
 *     proxy: "http://user:pass@1.2.3.4:8080",
 *   });
 *   const { objects } = await client.classify("aws:grid:bed", base64Tiles);
 *
 * Every API failure is thrown as CapsolverError with the CapSolver errorCode. Transport
 * failures use the client-side code ERROR_CLIENT_TRANSPORT, polling timeouts ERROR_CLIENT_TIMEOUT.
 */

export const ERROR_CLIENT_TIMEOUT = "ERROR_CLIENT_TIMEOUT";
export const ERROR_CLIENT_TRANSPORT = "ERROR_CLIENT_TRANSPORT";

/**
 * @typedef {{cookie: string, userAgent: string, captchaVoucher?: string}} WafSolution
 *   cookie is "" for a voucher-only outcome (CAPTCHA solved, no token host to redeem at)
 * @typedef {{objects: number[]}} GridSolution
 * @typedef {{box: [number, number]}} ToycarSolution
 */

export class CapsolverError extends Error {
  /**
   * @param {string} code CapSolver error code (ERROR_...)
   * @param {string} [description]
   * @param {string|null} [taskId]
   */
  constructor(code, description = "", taskId = null) {
    super(description ? `${code}: ${description}` : code);
    this.name = "CapsolverError";
    this.code = code;
    this.description = description;
    this.taskId = taskId;
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function toBase64(image) {
  if (typeof image === "string") return image;
  if (image instanceof Uint8Array || image instanceof ArrayBuffer) {
    const bytes = image instanceof ArrayBuffer ? new Uint8Array(image) : image;
    if (typeof Buffer !== "undefined") return Buffer.from(bytes).toString("base64");
    let bin = "";
    for (let i = 0; i < bytes.length; i += 0x8000) {
      bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    }
    return btoa(bin);
  }
  throw new TypeError("images must be base64 strings, Uint8Array/Buffer or ArrayBuffer");
}

export class AwsCapsolverClient {
  /**
   * @param {object} opts
   * @param {string} [opts.baseUrl="http://127.0.0.1:8790"]
   * @param {string} opts.clientKey
   * @param {number} [opts.requestTimeoutMs=90000] bound for every single HTTP call; must
   *   exceed the server's AWSCS_CLASSIFY_TIMEOUT (classification answers synchronously)
   */
  constructor({ baseUrl = "http://127.0.0.1:8790", clientKey = "", requestTimeoutMs = 90000 } = {}) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.clientKey = clientKey;
    this.requestTimeoutMs = requestTimeoutMs;
  }

  async #request(method, path, payload) {
    let resp;
    try {
      resp = await fetch(this.baseUrl + path, {
        method,
        headers: payload ? { "content-type": "application/json" } : undefined,
        body: payload ? JSON.stringify({ ...payload, clientKey: this.clientKey }) : undefined,
        signal: AbortSignal.timeout(this.requestTimeoutMs),
      });
    } catch (err) {
      throw new CapsolverError(ERROR_CLIENT_TRANSPORT, `${path}: ${err.message || err}`);
    }
    let text;
    try {
      text = await resp.text();
    } catch (err) {
      throw new CapsolverError(ERROR_CLIENT_TRANSPORT, `${path}: ${err.message || err}`);
    }
    let data = null;
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
    const isObject = data !== null && typeof data === "object" && !Array.isArray(data);
    if (!isObject || (resp.status !== 200 && !data.errorId)) {
      // a proxy error page or similar; the service itself always answers a JSON object
      throw new CapsolverError(ERROR_CLIENT_TRANSPORT, `${path}: HTTP ${resp.status} ${text.slice(0, 200)}`);
    }
    if (data.errorId) {
      // also for HTTP 500 replies that carry an errorCode
      throw new CapsolverError(data.errorCode || "ERROR_UNKNOWN", data.errorDescription || "", data.taskId ?? null);
    }
    return data;
  }

  /** POST /createTask; returns the raw reply ({taskId} or {taskId, status:"ready", solution}). */
  createTask(task) {
    return this.#request("POST", "/createTask", { task });
  }

  /**
   * Poll /getTaskResult until ready and return the solution.
   * @param {string} taskId
   * @param {{timeoutMs?: number, pollIntervalMs?: number}} [opts]
   */
  async getTaskResult(taskId, { timeoutMs = 150000, pollIntervalMs = 2000 } = {}) {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const data = await this.#request("POST", "/getTaskResult", { taskId });
      if (data.status === "ready") return data.solution || {};
      const remaining = deadline - Date.now();
      if (remaining <= 0) {
        throw new CapsolverError(ERROR_CLIENT_TIMEOUT, `task ${taskId} not ready after ${timeoutMs} ms`, taskId);
      }
      await sleep(Math.min(pollIntervalMs, remaining));
    }
  }

  getBalance() {
    return this.#request("POST", "/getBalance", {});
  }

  health() {
    return this.#request("GET", "/health");
  }

  /**
   * Solve the AWS WAF of a site. Uses AntiAwsWafTask with a proxy, otherwise the ProxyLess type.
   * @param {string} websiteURL
   * @param {{proxy?: string, userAgent?: string, timeoutMs?: number, pollIntervalMs?: number,
   *          extra?: Record<string, unknown>}} [opts] extra: awsKey, awsIv, awsContext, ...
   * @returns {Promise<WafSolution>}
   */
  async solveAwsWaf(websiteURL, { proxy, userAgent, timeoutMs, pollIntervalMs, extra = {} } = {}) {
    const task = { type: proxy ? "AntiAwsWafTask" : "AntiAwsWafTaskProxyLess", websiteURL, ...extra };
    if (proxy) task.proxy = proxy;
    if (userAgent) task.userAgent = userAgent;
    const created = await this.createTask(task);
    if (created.status === "ready") return created.solution || {};
    if (!created.taskId) throw new CapsolverError(ERROR_CLIENT_TRANSPORT, "createTask returned no taskId");
    return this.getTaskResult(created.taskId, { timeoutMs, pollIntervalMs });
  }

  /**
   * AwsWafClassification (synchronous).
   * @param {string} question "aws:grid:<target>" or "aws:toycarcity:carcity"
   * @param {Array<string|Uint8Array|ArrayBuffer>} images base64 strings or raw bytes
   * @returns {Promise<GridSolution|ToycarSolution>}
   */
  async classify(question, images) {
    const created = await this.createTask({
      type: "AwsWafClassification",
      question,
      images: images.map(toBase64),
    });
    if (created.status === "ready") return created.solution || {};
    return this.getTaskResult(created.taskId);
  }
}

export default AwsCapsolverClient;
