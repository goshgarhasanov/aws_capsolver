# aws_capsolver

A self-hosted AWS WAF solving service that uses the CapSolver API. A client that already
talks to CapSolver only needs a new `baseUrl` and `clientKey`.

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
  vision/         tile classifier / toycar / audio (separate component)
  vision_bridge.py lazy import of vision/, works without it
  tasks.py        task validation, bounded worker pool, registry with TTL
  server.py       FastAPI app
  config.py       env settings, client keys, SQLite usage counters
tools/e2e_waf.py  end-to-end check in Chrome against a WAF endpoint you control
```

## Run

```bash
echo '{"MY-KEY": {"name": "bot-a", "balance": 1000}}' > keys.json   # or AWSCS_KEYS=k1,k2
./run.sh                       # == .venv/bin/python -m aws_capsolver serve --port 8790
.venv/bin/python -m aws_capsolver solve https://protected.example/ --proxy host:port:user:pass
```

Environment variables, all optional:

| variable | default | meaning |
|---|---|---|
| `AWSCS_KEYS_FILE` | `keys.json` | client keys, either `["k1"]` or `{"k1": {"balance": 10}}`; reloaded when the file changes |
| `AWSCS_KEYS` | | comma-separated extra keys |
| `AWSCS_OPEN_ACCESS` | `0` | accept any key (local dev only; same as `serve --open`) |
| `AWSCS_USAGE_DB` | `data/usage.db` | SQLite per-key counters |
| `AWSCS_WORKERS` | `8` | concurrent solving tasks |
| `AWSCS_MAX_PENDING` | `200` | queued + running tasks before `ERROR_NO_SLOT_AVAILABLE` |
| `AWSCS_TASK_TIMEOUT` | `120` | hard limit per task, seconds |
| `AWSCS_TASK_TTL` | `600` | finished tasks are forgotten after this |
| `AWSCS_HTTP_TIMEOUT` | `20` | per network call |
| `AWSCS_CLASSIFY_TIMEOUT` | `60` | synchronous classification limit |
| `AWSCS_POW_PROCESSES` | CPU count | hashcash worker processes |
| `AWSCS_CAPTCHA_ROUNDS` | `5` | puzzles to attempt before giving up |
| `AWSCS_VERIFY_TOKEN` | `1` | re-fetch the page with the PoW token to detect a CAPTCHA action |
| `AWSCS_WARMUP` | `1` | preload the vision model at startup (first load takes about 40 s on CPU) |

## API

Every reply uses HTTP 200. Errors look like
`{"errorId": 1, "errorCode": "ERROR_...", "errorDescription": "..."}`.

### AntiAwsWafTask / AntiAwsWafTaskProxyLess

```bash
curl -s localhost:8790/createTask -d '{
  "clientKey": "MY-KEY",
  "task": {
    "type": "AntiAwsWafTask",
    "websiteURL": "https://protected.example/",
    "proxy": "http://user:pass@1.2.3.4:8080",
    "userAgent": "Mozilla/5.0 ..."
  }}'
# {"errorId":0,"taskId":"5f0c..."}

curl -s localhost:8790/getTaskResult -d '{"clientKey":"MY-KEY","taskId":"5f0c..."}'
# {"errorId":0,"taskId":"5f0c...","status":"processing"}
# {"errorId":0,"taskId":"5f0c...","status":"ready",
#  "solution":{"cookie":"<aws-waf-token>","userAgent":"Mozilla/5.0 ..."}}
```

Proxy formats: `scheme://user:pass@host:port`, `scheme:host:port:user:pass`,
`host:port[:user:pass]`, or the CapSolver fields `proxyType/proxyAddress/proxyPort/proxyLogin/proxyPassword`.

Optional inputs: `awsKey`, `awsIv`, `awsContext` (gokuProps), `awsChallengeJS`, `awsProblemUrl`,
`awsApiJs` + `awsApiKey` (jsapi widget), `locale`. If you leave them out, the service fetches
`websiteURL` through the proxy and extracts them. Send the cookie back with the same
User-Agent and from the same IP address.

Flow:
1. Run the PoW challenge on the token host to get a token.
2. If the page is a CAPTCHA interstitial (HTTP 405 with `captcha.js`), or the page still
   blocks when re-fetched with the PoW token, solve the visual puzzle. If it is unsolvable,
   fall back to the audio puzzle.
3. Exchange the voucher for the final token, passing `existing_token` = the PoW token.

### AwsWafClassification (synchronous)

```bash
curl -s localhost:8790/createTask -d '{"clientKey":"MY-KEY","task":{
  "type":"AwsWafClassification","question":"aws:grid:bed","images":["/9j/...", "..."]}}'
# {"errorId":0,"taskId":"...","status":"ready","solution":{"objects":[1,4,7]}}
# "aws:toycarcity:carcity" -> {"solution":{"box":[x,y]}}
```

### getBalance

`{"clientKey": "MY-KEY"}` returns `{"errorId":0,"balance":1000,"usage":{"AntiAwsWafTask":{"created":3,"solved":3,"failed":0}}}`.
The service takes no payments; `balance` is the static value from `keys.json`.

### Error codes

`ERROR_KEY_DENIED_ACCESS`, `ERROR_INVALID_TASK_DATA`, `ERROR_TASK_NOT_SUPPORTED`,
`ERROR_TASKID_INVALID`, `ERROR_CAPTCHA_UNSOLVABLE`, `ERROR_PROXY_CONNECT_REFUSED`,
`ERROR_PROXY_BANNED`, `ERROR_SITE_UNREACHABLE`, `ERROR_TASK_TIMEOUT`, `ERROR_NO_SLOT_AVAILABLE`,
`ERROR_SERVICE_UNAVAILABLE`.

## Protocol notes (checked against captcha.js / challenge.js, Nov 2026)

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

## Tests

```bash
.venv/bin/python -m pytest
```

The tests cover parsing, URL derivation, PoW against an independent reference search
(sequential and parallel), the full challenge, captcha and voucher exchange on a scripted
fake session, and the API contract through FastAPI's TestClient. They make no network calls.
