# Captchosa

**Self-hosted captcha solving service with a CapSolver-compatible API, plus a Chrome extension.**

Captchosa runs on your own machine or server. Any client that already talks to CapSolver
works after changing two values: `baseUrl` and `clientKey`. Nothing is sent to a third-party
solving service; the models run locally (CPU is enough).

| | |
| --- | --- |
| **Service** | Python / FastAPI, package `aws_capsolver`, default port `8790` |
| **Extension** | Chrome / Chromium, Manifest V3, v1.0.5 ([download](../../releases/latest)) |
| **API** | `createTask`, `getTaskResult`, `getBalance`, `health` (CapSolver-style) |
| **Runs on** | Linux / macOS, Python 3.10+, CPU-only PyTorch, or Docker |

---

## What it solves

| Task type | Captcha | Notes |
| --- | --- | --- |
| `AntiAwsWafTask` / `AntiAwsWafTaskProxyLess` | **AWS WAF** challenge + captcha | Full flow: PoW challenge, visual grid puzzle (audio fallback), returns the `aws-waf-token` cookie |
| `AwsWafClassification` | AWS WAF grid ("choose all the beds") | One already-fetched problem, returns the tile indexes |
| `ImageToTextTask` | Text / image captchas | BLS Poland model, a general text model, ddddocr fallback |
| `GeeTestClassification` | GeeTest v3 / v4 pictures | Slide gap, icon / word / space click points, nine-grid, gobang |
| `BlsClassification` | BLS "select all boxes with number NNN" grid | Needs the optional `bls_capsolver` model |
| `ReCaptchaV2Classification` | reCAPTCHA v2 image challenge (3x3, 4x4, dynamic 1x1 tiles) | One already-fetched picture + the object (`/m/0k4j` or text in any language); returns the cells to click. No tokens are produced |

### Measured results

| What | Result |
| --- | --- |
| AWS WAF token (live site, `AntiAwsWafTaskProxyLess`, 2 CPU server) | 10 / 10 tokens accepted by the WAF, **~1.4 s** average (1.0 - 1.8 s) |
| AWS WAF grid head (ViT-B-32) | 319 / 319 on the held-out grid set, ~125 ms per grid |
| GeeTest via the extension (live demo pages) | v3 4 / 4, v4 6 / 6 |

---

## Browser extension

The extension finds captchas on the page, sends them **only to your own Captchosa URL** and
applies the answer like a person would (pointer events, human-like slider drags).

| Popup (light) | Popup (dark) |
| --- | --- |
| ![Popup, light theme](extension/docs/screenshots/popup-light.png) | ![Popup, dark theme](extension/docs/screenshots/popup-dark.png) |

![Options page](extension/docs/screenshots/options-light.png)

Supported on the page: AWS WAF grid widget and token pages, GeeTest v3 / v4, the reCAPTCHA v2
image challenge, BLS grid and BLS Poland text captchas, and general text captchas next to a captcha input.
UI languages: English, Turkish, Azerbaijani, Russian.

### Install

1. Download `captchosa-extension-<version>.zip` from the [latest release](../../releases/latest) and unzip it.
2. Open `chrome://extensions`, turn on **Developer mode**.
3. Click **Load unpacked** and pick the unzipped folder.
4. Open the extension options, enter your service URL (e.g. `http://127.0.0.1:8790`) and your client key.

Build the zip yourself: `python3 extension/tools/build_extension.py` -> `extension/dist/`.
Details: [extension/README.md](extension/README.md) - changes: [extension/CHANGELOG.md](extension/CHANGELOG.md).

---

## Quick start (service)

```bash
git clone https://github.com/goshgarhasanov/aws_capsolver.git captchosa
cd captchosa

# venv + CPU-only torch + dependencies + keys.json with a random key (+ optional systemd unit)
deploy/install.sh

./run.sh                               # serves on 127.0.0.1:8790
curl -s localhost:8790/health          # {"ok":true,"version":"0.1.0","vision":true,...}
```

Or with Docker:

```bash
docker compose up -d
```

Client keys live in `keys.json` (git-ignored):

```json
{ "MY-KEY": { "name": "bot-a", "balance": 1000000 } }
```

> Every key has a **token balance**; each solved task costs tokens (AWS WAF token = 30,
> grid = 10, GeeTest = 8, BLS grid = 5, reCAPTCHA v2 picture = 3, text = 2). Failed tasks are refunded.
> A key without `"balance"` starts with only 1000 tokens, so set a high balance for your own keys.

---

## Usage

### curl (CapSolver-style)

```bash
curl -s localhost:8790/createTask -d '{
  "clientKey": "MY-KEY",
  "task": { "type": "AntiAwsWafTaskProxyLess", "websiteURL": "https://protected.example/" }
}'
# -> {"errorId":0,"taskId":"5f0c2b8e-..."}

curl -s localhost:8790/getTaskResult -d '{"clientKey":"MY-KEY","taskId":"5f0c2b8e-..."}'
# -> {"errorId":0,"status":"ready","solution":{"cookie":"...","userAgent":"..."}}
```

### Python client

```python
# copy clients/python/aws_capsolver_client.py next to your code (depends only on requests)
from aws_capsolver_client import AwsCapsolverClient

client = AwsCapsolverClient("http://127.0.0.1:8790", client_key="MY-KEY")
solution = client.solve_aws_waf("https://protected.example/", proxy="http://user:pass@1.2.3.4:8080")
print(solution["cookie"])          # aws-waf-token value
print(client.get_balance())
```

A dependency-free JavaScript client is in [`clients/js/`](clients/js/awsCapsolverClient.js).

### Existing CapSolver clients

Point the SDK's API base URL at your server (e.g. `http://127.0.0.1:8790`) and use your own
client key; the request and response formats are the same.

---

## Project layout

```
aws_capsolver/   the service: waf/ (AWS WAF protocol), vision/ (grid models), text/ (OCR),
                 geetest/, recaptcha/, tasks.py, billing.py, server.py
extension/       Chrome extension (MV3) + build tools
clients/         Python and JavaScript clients
deploy/          install.sh + systemd unit
tools/           load test, end-to-end checks, dataset / training / benchmark tools
weights/         trained model heads (backbones are downloaded on first start)
tests/           pytest suite (run in CI)
docs/            full reference
```

## Documentation

The full reference is in **[docs/REFERENCE.md](docs/REFERENCE.md)**: every task type and field,
error codes, configuration (`AWSCS_*` environment variables), billing, deployment, load test,
known limits, retraining the models and protocol notes.

## Tests

```bash
.venv/bin/python -m pytest
```

## Responsible use

Use Captchosa only on sites you own or are allowed to automate, and follow their terms of
service. The project is provided as is, without warranty.
