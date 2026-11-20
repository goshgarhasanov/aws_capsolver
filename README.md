# captchosa (package `aws_capsolver`)

A self-hosted AWS WAF and BLS captcha solving service that speaks the CapSolver API. A client that already
talks to CapSolver only needs a new `baseUrl` and `clientKey`.

- `AntiAwsWafTask` / `AntiAwsWafTaskProxyLess`: solve a site's AWS WAF (PoW challenge,
  then the visual or audio CAPTCHA when needed) and return the `aws-waf-token` cookie.
- `AwsWafClassification`: answer one already-fetched CAPTCHA problem (grid tiles).
- `BlsClassification`: the BLS "select all boxes with number NNN" grid (bls_capsolver CNN).
- `getBalance`, `getTaskResult`, `/health`.

Contents: [Layout](#layout) - [Quick start](#quick-start) - [API reference](#api-reference) -
[Error codes](#error-codes) - [Configuration](#configuration) - [Clients](#clients) -
[Deployment](#deployment) - [Load test](#load-test) - [Known limits](#known-limits) - [Retraining the models](#retraining-the-models) -
[Vision notes](#vision-notes) - [Protocol notes](#protocol-notes) - [Tests and CI](#tests-and-ci)

## Layout

```
aws_capsolver/
  waf/            protocol layer (no web framework)
    urls.py       token host <-> captcha host derivation
    parse.py      gokuProps / challenge.js / captcha.js / jsapi.js + apiKey extraction
    pow.py        sha256 + scrypt hashcash (multi-process), NetworkBandwidth payload
    challenge.py  /inputs -> solve -> /verify | /mp_verify -> token
    captcha.py    /problem -> vision -> /verify (retry / multi-round) -> /voucher -> token
    solver.py     whole-site flow: page -> PoW -> token check -> captcha when needed
    http.py       sessions, proxy parsing, network error -> error code mapping
  vision/         grid tile classifier (CLIP), toycar stub, audio (faster-whisper)
  vision_bridge.py lazy import of vision/, works without it
  tasks.py        task validation, bounded worker pool, registry with TTL
  server.py       FastAPI app
  config.py       env settings, client keys, SQLite usage counters
clients/python/   aws_capsolver_client.py - typed client, depends only on requests
clients/js/       awsCapsolverClient.js   - fetch-based ES module, no dependencies
deploy/           systemd unit + install.sh
tools/            loadtest.py, e2e_waf.py, dataset / training / benchmark tools
weights/          aws_capsolver_head.pt (trained head; backbones are downloaded on demand)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Quick start

### Local (virtualenv)

```bash
deploy/install.sh --no-deps        # only creates keys.json with a random key (skip pip)
# or the full install: venv + CPU-only torch + dependencies + keys.json + systemd unit
deploy/install.sh

# or by hand:
echo '{"MY-KEY": {"name": "bot-a", "balance": 1000}}' > keys.json   # or AWSCS_KEYS=k1,k2
./run.sh                       # == .venv/bin/python -m aws_capsolver serve --port 8790
curl -s localhost:8790/health  # {"ok":true,"version":"0.1.0","vision":true,"time":...}

# one-off solve from the command line
.venv/bin/python -m aws_capsolver solve https://protected.example/ --proxy host:port:user:pass
```

`serve` options: `--host` (default `127.0.0.1`), `--port` (default `8790`), `--open` (accept
any `clientKey`, development only), and the global `--log-level`.

The first classification loads the CLIP backbones (about 40 s on CPU, longer when they still
have to be downloaded into the Hugging Face cache). With `AWSCS_WARMUP=1` (the default)
this happens in the background at startup.

### Docker

```bash
mkdir -p config && echo '{"MY-KEY": {"balance": 1000}}' > config/keys.json
docker compose up -d --build
curl -s localhost:8790/health
```

See [Deployment](#deployment) for details.

## API reference

All endpoints take and return JSON. Replies use HTTP 200 (like CapSolver), including
errors, which carry `errorId: 1`. The only exception is an unexpected server bug: HTTP 500
with `ERROR_SERVICE_UNAVAILABLE` / `internal error`.

```json
{"errorId": 1, "errorCode": "ERROR_...", "errorDescription": "..."}
```

Every POST body must be a JSON object with `clientKey`, at most `AWSCS_MAX_BODY_BYTES`
(4 MB) long. `/docs`, `/redoc` and `/openapi.json` are disabled; this section is the
reference.

Targets are checked against an SSRF guard that is on by default: a `websiteURL` (or WAF
endpoint) that resolves to a private, loopback or reserved address is rejected with
`ERROR_INVALID_TASK_DATA`. Set `AWSCS_ALLOW_PRIVATE=1` to allow such targets (testing
against a local WAF setup, for example).

### POST /createTask - AntiAwsWafTask / AntiAwsWafTaskProxyLess (asynchronous)

```bash
curl -s localhost:8790/createTask -d '{
  "clientKey": "MY-KEY",
  "task": {
    "type": "AntiAwsWafTask",
    "websiteURL": "https://protected.example/",
    "proxy": "http://user:pass@1.2.3.4:8080",
    "userAgent": "Mozilla/5.0 ..."
  }}'
```
```json
{"errorId": 0, "taskId": "5f0c2b8e-..."}
```

| field | required | meaning |
|---|---|---|
| `type` | yes | `AntiAwsWafTask` (needs a proxy) or `AntiAwsWafTaskProxyLess` (direct from the server) |
| `websiteURL` | yes | page protected by AWS WAF |
| `proxy` | `AntiAwsWafTask` | `scheme://user:pass@host:port`, `scheme:host:port:user:pass`, `host:port[:user:pass]` |
| `proxyType`, `proxyAddress`, `proxyPort`, `proxyLogin`, `proxyPassword` | alternative | CapSolver-style proxy fields instead of `proxy` |
| `userAgent` | no | UA used for every request; send the cookie with the same UA |
| `awsKey`, `awsIv`, `awsContext` | no | `gokuProps` from the page |
| `awsChallengeJS` | no | URL of `challenge.js` |
| `awsProblemUrl` | no | captcha `/problem` URL |
| `awsApiJs`, `awsApiKey` | no | jsapi widget script and its API key |
| `locale` | no | captcha locale, default `en-us` |

If the optional `aws*` inputs are left out, the service fetches `websiteURL` through the
proxy and extracts them. Use the cookie from the same IP address (the same proxy) and the
same User-Agent.

Flow:
1. Run the PoW challenge on the token host to get a token.
2. If the page is a CAPTCHA interstitial (HTTP 405 with `captcha.js`), or the page still
   blocks when re-fetched with the PoW token, solve the visual puzzle. If it is unsolvable,
   fall back to the audio puzzle.
3. Exchange the voucher for the final token, passing `existing_token` = the PoW token.

### POST /getTaskResult

```bash
curl -s localhost:8790/getTaskResult -d '{"clientKey":"MY-KEY","taskId":"5f0c2b8e-..."}'
```
While running:
```json
{"errorId": 0, "taskId": "5f0c2b8e-...", "status": "processing"}
```
Solved:
```json
{"errorId": 0, "taskId": "5f0c2b8e-...", "status": "ready",
 "solution": {"cookie": "<aws-waf-token>", "userAgent": "Mozilla/5.0 ...",
              "captchaVoucher": "<only when a CAPTCHA was solved>"}}
```
Voucher-only outcome: when a CAPTCHA was solved but there is no token host to redeem the
voucher at (for example a jsapi widget page without `challenge.js`), the task is still
`ready`, with `"cookie": ""` and the `captchaVoucher`. Hand the voucher to the page the way
the widget would (its `renderCaptcha` callback / the site's own form) instead of setting a
cookie. Always check `cookie` before using it.

Failed (the task's error code; see [Error codes](#error-codes)):
```json
{"errorId": 1, "errorCode": "ERROR_CAPTCHA_UNSOLVABLE", "errorDescription": "...",
 "taskId": "5f0c2b8e-...", "status": "failed"}
```

Poll every 2-3 s. A task is never `processing` for longer than `AWSCS_TASK_TIMEOUT` + 15 s
(it then fails with `ERROR_TASK_TIMEOUT`); finished tasks are forgotten after
`AWSCS_TASK_TTL` seconds (`ERROR_TASKID_INVALID` afterwards). A task is visible only to
the `clientKey` that created it.

### POST /createTask - AwsWafClassification (synchronous)

Answers in the `createTask` reply; no polling needed.

Grid problem (`question` = `aws:grid:<target>`, `images` = the base64 tiles in row-major
order, as in the `/problem` reply's `assets.images`; `data:` URLs are accepted):

```bash
curl -s localhost:8790/createTask -d '{"clientKey":"MY-KEY","task":{
  "type":"AwsWafClassification","question":"aws:grid:bed","images":["/9j/...", "..."]}}'
```
```json
{"errorId": 0, "taskId": "...", "status": "ready", "solution": {"objects": [1, 4, 5, 7, 8]}}
```

`objects` are 0-based tile indices. Limits: at most 16 images of 2 MB each, and the whole
body within `AWSCS_MAX_BODY_BYTES`.

Toycar problem (`question` = `aws:toycarcity:carcity`, `images` = one image):
```json
{"errorId": 0, "taskId": "...", "status": "ready", "solution": {"box": [x, y]}}
```
No toycar problem has been observed on the sampled endpoints, so this currently answers
`ERROR_CAPTCHA_UNSOLVABLE`.

### POST /createTask - BlsClassification (synchronous)

The BLS grid captcha (turkey.blsspainglobal.com and other BLS sites): "select all boxes
with number NNN". Send the 9 **visible** tiles in page order (BLS also renders hidden
decoy tiles; leave them out) and the target from `question` (`bls:grid:<NNN>`) or `target`.

```bash
curl -s localhost:8790/createTask -H 'content-type: application/json' -d '{"clientKey":"K","task":{
  "type":"BlsClassification","question":"bls:grid:748","images":["iVBOR...", "..."]}}'
```
```json
{"errorId": 0, "taskId": "...", "status": "ready",
 "solution": {"objects": [1, 3, 4], "readings": ["728", "748", "658", "748", "748", "986", "351", "860", "498"]}}
```
`objects` are the tile indices to click; `readings` is what the model read on each tile
(useful for logging). Errors: `ERROR_INVALID_TASK_DATA` (target not 3 digits, not exactly 9
tiles, undecodable tile), `ERROR_CAPTCHA_UNSOLVABLE` (no tile matches),
`ERROR_TASK_NOT_SUPPORTED` (bls_capsolver not installed). It shares the classification
queue and limits with `AwsWafClassification`.

The model is the separate private project `goshgarhasanov/bls_capsolver`, installed into
the same environment:
```bash
git clone https://github.com/goshgarhasanov/bls_capsolver ../bls_capsolver
.venv/bin/pip install --no-deps -e ../bls_capsolver
```
The decision threshold defaults to 0.3 (as in the BlsSpainTurkey bot) and can be changed
with `BLS_CAPSOLVER_THRESHOLD`; `BLS_CAPSOLVER_WEIGHTS` points at another checkpoint.

### POST /getBalance

```bash
curl -s localhost:8790/getBalance -d '{"clientKey":"MY-KEY"}'
```
```json
{"errorId": 0, "balance": 1000.0, "packages": [],
 "usage": {"AntiAwsWafTask": {"created": 3, "solved": 3, "failed": 0},
           "AwsWafClassification": {"created": 10, "solved": 10, "failed": 0}}}
```
The service takes no payments; `balance` is the static value from `keys.json`
(default 1000).

### GET /health

```json
{"ok": true, "version": "0.1.0", "vision": true, "time": 1795127465}
```
`ok` turns `false` while the service is shutting down. `vision` is `false` when the vision
package cannot be imported; WAF tasks then still solve
PoW-only pages but every CAPTCHA fails with `ERROR_CAPTCHA_UNSOLVABLE`. Used by the Docker
healthcheck.

## Error codes

| errorCode | returned by | when |
|---|---|---|
| `ERROR_KEY_DENIED_ACCESS` | every POST | `clientKey` missing (`clientKey is required`) or not in `keys.json` / `AWSCS_KEYS` (`invalid clientKey`) |
| `ERROR_INVALID_TASK_DATA` | every POST / createTask | body is not a JSON object or exceeds `AWSCS_MAX_BODY_BYTES`; no `task`; missing or non-http(s) `websiteURL`; over-long fields or control characters; `AntiAwsWafTask` without a proxy; unparsable proxy; non-string optional fields; classification without `question`/`images`, bad question format, more than 16 images, bad base64 or undecodable image; target URL resolves to a private address (unless `AWSCS_ALLOW_PRIVATE=1`) |
| `ERROR_TASK_NOT_SUPPORTED` | createTask | unknown `task.type`, or a classification question family other than `grid` / `toycarcity` |
| `ERROR_TASKID_INVALID` | getTaskResult | unknown, expired, or another key's `taskId` |
| `ERROR_CAPTCHA_UNSOLVABLE` | task result / classification | every CAPTCHA round failed, unsupported problem type, vision unavailable, or an internal solver error |
| `ERROR_PROXY_CONNECT_REFUSED` | task result | the proxy failed, refused the connection, timed out connecting, or answered 407 (proxy authentication) |
| `ERROR_PROXY_BANNED` | task result | a WAF endpoint answered HTTP 403 / 429 through the proxy (the proxy IP is blocked or rate limited) |
| `ERROR_SITE_UNREACHABLE` | task result | DNS failure, connection or TLS error (without a proxy), too many redirects, oversized or unexpected reply from the site or the WAF endpoints |
| `ERROR_TASK_TIMEOUT` | task result / classification | the task exceeded `AWSCS_TASK_TIMEOUT`, a network call timed out, or classification exceeded `AWSCS_CLASSIFY_TIMEOUT` |
| `ERROR_NO_SLOT_AVAILABLE` | createTask | `AWSCS_MAX_PENDING` tasks are already queued or running; retry later |
| `ERROR_SERVICE_UNAVAILABLE` | createTask / any | the service is shutting down; with HTTP 500, an unexpected internal error |

Example of each shape:

```json
{"errorId": 1, "errorCode": "ERROR_KEY_DENIED_ACCESS", "errorDescription": "invalid clientKey"}
{"errorId": 1, "errorCode": "ERROR_INVALID_TASK_DATA", "errorDescription": "question must look like aws:grid:<target>"}
{"errorId": 1, "errorCode": "ERROR_TASK_NOT_SUPPORTED", "errorDescription": "task type 'Foo' is not supported"}
{"errorId": 1, "errorCode": "ERROR_TASKID_INVALID", "errorDescription": "task not found or expired"}
{"errorId": 1, "errorCode": "ERROR_CAPTCHA_UNSOLVABLE", "errorDescription": "toycarcity problems are not supported yet", "taskId": "...", "status": "failed"}
{"errorId": 1, "errorCode": "ERROR_PROXY_CONNECT_REFUSED", "errorDescription": "...", "taskId": "...", "status": "failed"}
{"errorId": 1, "errorCode": "ERROR_PROXY_BANNED", "errorDescription": "...", "taskId": "...", "status": "failed"}
{"errorId": 1, "errorCode": "ERROR_SITE_UNREACHABLE", "errorDescription": "website: connection failed", "taskId": "...", "status": "failed"}
{"errorId": 1, "errorCode": "ERROR_TASK_TIMEOUT", "errorDescription": "task exceeded the time limit", "taskId": "...", "status": "failed"}
{"errorId": 1, "errorCode": "ERROR_NO_SLOT_AVAILABLE", "errorDescription": "too many tasks in progress"}
{"errorId": 1, "errorCode": "ERROR_SERVICE_UNAVAILABLE", "errorDescription": "service is shutting down"}
```

The bundled clients add two client-side codes: `ERROR_CLIENT_TRANSPORT` (connection
failure, request timeout, or a reply that is not the service's JSON) and `ERROR_CLIENT_TIMEOUT` (polling gave up).

## Configuration

Service settings (all optional; read at startup, except the keys file which is reloaded
when it changes):

| variable | default | meaning |
|---|---|---|
| `AWSCS_KEYS_FILE` | `keys.json` | client keys, either `["k1"]` or `{"k1": {"name": "bot-a", "balance": 10}}` |
| `AWSCS_KEYS` | | comma-separated extra keys |
| `AWSCS_OPEN_ACCESS` | `0` | accept any key (local dev only; same as `serve --open`) |
| `AWSCS_USAGE_DB` | `data/usage.db` | SQLite per-key counters |
| `AWSCS_WORKERS` | `8` | concurrent WAF tasks (classification has its own runners, see `AWSCS_VISION_CONCURRENCY`) |
| `AWSCS_MAX_PENDING` | `200` | queued + running tasks before `ERROR_NO_SLOT_AVAILABLE` |
| `AWSCS_TASK_TIMEOUT` | `120` | hard limit per WAF task, seconds |
| `AWSCS_TASK_TTL` | `600` | finished tasks are forgotten after this |
| `AWSCS_HTTP_TIMEOUT` | `20` | per network call |
| `AWSCS_CLASSIFY_TIMEOUT` | `60` | synchronous classification limit |
| `AWSCS_VISION_CONCURRENCY` | `1` | classifications running at once on the dedicated vision executor (the model serialises calls, so more rarely helps) |
| `AWSCS_MAX_CLASSIFY_QUEUE` | `8 × AWSCS_VISION_CONCURRENCY` | classifications allowed to wait; beyond running + waiting, `createTask` answers `ERROR_NO_SLOT_AVAILABLE` at once. A request that expires while waiting is never run |
| `AWSCS_POW_PROCESSES` | CPU count | hashcash worker processes |
| `AWSCS_CAPTCHA_ROUNDS` | `5` | puzzles to attempt before giving up |
| `AWSCS_VERIFY_TOKEN` | `1` | re-fetch the page with the PoW token to detect a CAPTCHA action |
| `AWSCS_WARMUP` | `1` | preload the vision model at startup |
| `AWSCS_MAX_BODY_BYTES` | `4194304` | request body limit |
| `AWSCS_ALLOW_PRIVATE` | `0` | allow `websiteURL` / WAF hosts that resolve to private addresses (SSRF guard off) |
| `AWSCS_LOG_LEVEL` | `INFO` | log level (same as `--log-level`) |

Vision settings:

| variable | default | meaning |
|---|---|---|
| `AWS_CAPSOLVER_CLIP_MODELS` | `ViT-B-32:laion2b_s34b_b79k,ViT-B-16:datacomp_xl_s13b_b90k` | open_clip backbones, `model:pretrained` comma-separated |
| `AWS_CAPSOLVER_HEAD` | `weights/aws_capsolver_head.pt` | trained head; `none` = zero-shot only |
| `AWS_CAPSOLVER_TORCH_THREADS` | `min(8, cpu_count)` | torch intra-op threads |
| `AWS_CAPSOLVER_TTA` | `0` | also embed the mirrored tile and average |
| `AWS_CAPSOLVER_GRID_K` | `5` | tiles to select on a 9-tile grid; `0` = head threshold |
| `AWS_CAPSOLVER_STRICT_TARGETS` | `0` | unknown targets fail instead of falling back to zero-shot |
| `AWS_CAPSOLVER_AUDIO_MODEL` | `small.en` | faster-whisper model for the audio CAPTCHA |
| `AWS_CAPSOLVER_AUDIO_THREADS` | `8` | CTranslate2 CPU threads |
| `HF_HOME` | `~/.cache/huggingface` | where backbones are cached (`/models/huggingface` in Docker) |
| `HF_HUB_OFFLINE` | `0` | `1` = never download; use once the cache is populated |

## Clients

Both clients are single files with no dependencies beyond `requests` (Python) or the
built-in `fetch` (Node >= 18 / browsers). Errors are raised as `CapsolverError` with
`.code` = the `errorCode`.

```python
from aws_capsolver_client import AwsCapsolverClient, CapsolverError   # clients/python/

client = AwsCapsolverClient("http://127.0.0.1:8790", "MY-KEY")
sol = client.solve_aws_waf("https://protected.example/", proxy="http://u:p@1.2.3.4:8080",
                           timeout=150)          # polls getTaskResult every 2 s
sol["cookie"], sol["userAgent"]
client.classify("aws:grid:bed", tiles_b64_or_bytes)  # {"objects": [...]}
client.create_task({...}); client.get_task_result(task_id, timeout=150, poll_interval=2)
client.get_balance(); client.health()
```

```js
import { AwsCapsolverClient, CapsolverError } from "./clients/js/awsCapsolverClient.js";

const client = new AwsCapsolverClient({ baseUrl: "http://127.0.0.1:8790", clientKey: "MY-KEY" });
const { cookie, userAgent } = await client.solveAwsWaf("https://protected.example/", {
  proxy: "http://u:p@1.2.3.4:8080", timeoutMs: 150000,
});
const { objects } = await client.classify("aws:grid:bed", tiles);   // base64 or Uint8Array
await client.createTask({...}); await client.getTaskResult(taskId, { timeoutMs, pollIntervalMs });
await client.getBalance(); await client.health();
```

Existing CapSolver SDKs also work: point their API base URL at this service.

## Deployment

The service has no TLS and no rate limiting of its own. Bind it to `127.0.0.1` (the default)
or put it behind a reverse proxy / firewall, and keep `keys.json` private.

Resources: plan for about 3 GB RAM per instance (2.8 GB RSS was measured with the default
backbones loaded after the load test below); the audio model adds to that when used.
Classification is CPU-bound and effectively serialised per process (see [Load test](#load-test)),
so throughput scales with faster CPUs or more instances, not with `AWSCS_WORKERS`.
Run exactly ONE server process per instance and scale WAF-task concurrency with
`AWSCS_WORKERS`. Never start uvicorn with `--workers N` (or several processes on one port):
the task registry is in memory, so `getTaskResult` would land on a process that does not
know the task. The Dockerfile, compose file and systemd unit all start a single process. If
you run several instances behind a load balancer, `getTaskResult` must reach the instance
that created the task (sticky routing per client, or one instance per client group).

### Docker / Compose

- Image: `python:3.12-slim`, CPU-only torch from the PyTorch CPU index, ffmpeg, runs as the
  non-root user `app` (uid 10001) under `tini`, listens on `0.0.0.0:8790` inside the container.
- Volumes: `/models` (Hugging Face / open_clip / whisper caches - downloaded on first use,
  so keep this volume), `/state` (`usage.db`), `/config` (`keys.json`, read-only).
- The trained head is baked into `/app/weights`; mount a directory over it to try a new one.
- `HEALTHCHECK` calls `/health` every 30 s.

```bash
docker build -t aws-capsolver .
docker run -d --name awscs -p 127.0.0.1:8790:8790 \
  -v awscs-models:/models -v awscs-state:/state -v "$PWD/config:/config:ro" \
  aws-capsolver
# or: docker compose up -d --build   (reads AWSCS_KEYS, AWSCS_WORKERS, ... from .env)
```

After the first successful classification the backbones are in the volume; set
`HF_HUB_OFFLINE=1` to stop the hub lookups at startup.

### systemd

`deploy/install.sh` (no root needed) creates `.venv`, installs CPU-only torch and the
dependencies, creates `keys.json` with a random key if there is none, and renders
`deploy/aws-capsolver.service` for this checkout and user into
`deploy/aws-capsolver.rendered.service`. It then prints the root steps instead of running them:

```bash
sudo install -m 0644 deploy/aws-capsolver.rendered.service /etc/systemd/system/aws-capsolver.service
sudo touch /etc/default/aws-capsolver        # optional AWSCS_* overrides, one per line
sudo systemctl daemon-reload
sudo systemctl enable --now aws-capsolver
journalctl -u aws-capsolver -f
```

`deploy/install.sh --user-unit` instead installs and starts a `systemctl --user` unit
(overrides in `~/.config/aws-capsolver/env`); `sudo loginctl enable-linger $USER` makes it
start at boot. The unit restarts on failure (5 s delay, at most 5 times in 10 min), binds
`AWSCS_HOST:AWSCS_PORT` (default `127.0.0.1:8790`) and runs `.venv/bin/python -m aws_capsolver serve`.
Installer options: `--no-deps`, and the environment variables `PYTHON`, `HOST`, `PORT`,
`TORCH_INDEX_URL`, `WITH_AUDIO=0`.

## Load test

`tools/loadtest.py` sends `AwsWafClassification` tasks built from the stored problem
`data/samples/problem_visual.json` to a running local server (it refuses non-local URLs
unless `--allow-remote`) and reports p50 / p95 latency and throughput per concurrency level.

```bash
.venv/bin/python -m aws_capsolver serve --open &
.venv/bin/python tools/loadtest.py                    # levels 1,4,16; 4 requests per worker
.venv/bin/python tools/loadtest.py --levels 1,8 --requests 50 --json
```

Measured 2026-11-20 on the development machine (32 vCPUs, heavily shared with other jobs:
load average 26-48 during the runs; default backbones, no head file, warm model). Two runs,
before and after the service hardening commit:

| concurrency | requests | p50 | p95 | throughput | errors |
|---|---|---|---|---|---|
| 1 | 4 | 2.10 s / 3.69 s | 2.25 s / 3.86 s | 0.48 / 0.28 req/s | - |
| 4 | 16 | 8.97 s / 13.7 s | 9.12 s / 14.4 s | 0.45 / 0.29 req/s | - |
| 16 | 64 | 32.7 s / 49.5 s | 40.8 s / 59.1 s | 0.47 / 0.10 req/s | 0 / 40 x `ERROR_TASK_TIMEOUT` |

Throughput is flat: classifications run one at a time, so latency grows linearly with the
queue. Once the queue wait exceeds `AWSCS_CLASSIFY_TIMEOUT` (60 s) requests fail with
`ERROR_TASK_TIMEOUT`, and the timed-out classifications still occupy the CPU, which drags
down the rest. Keep concurrent classification clients per instance at about
`AWSCS_CLASSIFY_TIMEOUT / single-request latency` (for example at most 8-10 here), or
add instances. Re-measure on the production host with `tools/loadtest.py`.

## Known limits

- The SSRF guard resolves the host before the request is made, and `requests` resolves it
  again when connecting. A hostile DNS server can answer a public address to the check
  and a private one to the connection (DNS rebinding). If that matters, run the service
  where it cannot reach internal networks (firewall egress, separate network namespace,
  or always solve through a proxy).
- Timed-out classifications are abandoned, not killed: the client gets
  `ERROR_TASK_TIMEOUT`, but the model call keeps running to completion in its thread and
  still uses CPU. Sustained overload therefore makes later requests slower (see
  [Load test](#load-test)). The same applies to a WAF task that passes
  `AWSCS_TASK_TIMEOUT`; it stops at its next deadline check.
- Tasks live in memory: a restart forgets every pending and finished task.
- `toycarcity` problems are not supported (no samples observed).
- No TLS, rate limiting or billing; `balance` is a static number.

## Retraining the models

The grid classifier is frozen CLIP backbones plus a small trained head
(`weights/aws_capsolver_head.pt`, logistic regression over the concatenated embeddings).
Retraining only touches the head:

```bash
# 1. collect problems (read-only GET /problem from an approved endpoint) -> data/grids/<id>/
.venv/bin/python tools/collect_samples.py --max 300 --delay 3.5
# 2. label: predict -> render review sheets -> accept each sheet with corrections
.venv/bin/python tools/label_tool.py predict
.venv/bin/python tools/label_tool.py sheets
.venv/bin/python tools/label_tool.py accept SHEET [pos=class ...]
# 3. train the head on the train split, check the held-out split
.venv/bin/python tools/train_head.py [--models A:B,C:D] [--tta] [--l2 1e-3]
.venv/bin/python tools/bench_vision.py                  # per-tile / per-grid accuracy + latency
# 4. final head on all splits (only after the held-out numbers are recorded)
.venv/bin/python tools/train_head.py --all
```

The head is picked up automatically when its backbones / TTA setting match the engine's
(`AWS_CAPSOLVER_CLIP_MODELS`, `AWS_CAPSOLVER_TTA`). In Docker, mount the new `weights/`
over `/app/weights` or rebuild the image. Audio: `tools/collect_audio.py` collects clips,
`tools/bench_audio.py transcribe | label | eval` labels them by multi-model agreement and
measures the solver on the frozen held-out split.

## Vision notes

<!-- VISION-NOTES: maintained by the vision component owners. Replace this block with
     model choices, dataset size, held-out accuracy, latency and known limitations. -->

_To be filled in by the vision component: backbones and head, dataset and splits, held-out
accuracy per class, latency per grid, audio solver accuracy, known failure modes._

<!-- /VISION-NOTES -->

## Protocol notes

Checked against captcha.js / challenge.js, Nov 2026.

- `/inputs` no longer returns `challenge.hmac`. The input is `uuid:...:blob`. The hashcash
  formula `sha256(input + hmac + counter)` treats a missing hmac as `""`.
- challenge.js dispatches on the hashed `challenge_type`, and the hash-to-solver mapping
  changes between builds. In the current build, `ha9faaff…` is NetworkBandwidth; in older
  builds it was scrypt. `challenge.py` therefore keeps an ordered list of candidate solvers
  for each hash and moves to the next one when `/verify` returns HTTP 400.
- captcha `/verify` is posted as `text/plain` to `/verify?region=<problem.region>`. The body is
  `{state,key,hmac_tag,client_solution,metrics:{solve_time_millis},goku_props,[api_key],locale}`.
  - `success:true` + `captcha_voucher`: done.
  - `success:true` + `problem`: more problems are required.
  - `success:false` + `problem`: wrong answer; the next attempt uses the returned problem.
- `client_solution` per problem family:
  - grid: sorted tile indices.
  - toycarcity: `{x:(2px-16)//32, y:(2py)//32}`, where `(px, py)` is a point in natural image pixels.
  - noisyneighbour: the text, lower-cased, trimmed and NFC-normalised.
- The voucher is exchanged at `<token>/voucher?region=` with `{captcha_voucher, existing_token}`.

## Tests and CI

```bash
.venv/bin/python -m pytest
AWS_CAPSOLVER_SKIP_MODEL_TESTS=1 .venv/bin/python -m pytest   # skip tests that load backbones
```

The tests cover parsing, URL derivation, PoW against an independent reference search
(sequential and parallel), the full challenge, captcha and voucher exchange on a scripted
fake session, the API contract through FastAPI's TestClient, and the vision / audio input
handling. They make no network calls. Tests that need model weights skip themselves when
the weights are not cached locally.

`.github/workflows/ci.yml` runs on every push: Python 3.12, CPU-only torch, pip cache,
`AWS_CAPSOLVER_SKIP_MODEL_TESTS=1` + `HF_HUB_OFFLINE=1`, the full test suite, then a smoke
test that starts the server and calls it with the Python client.
