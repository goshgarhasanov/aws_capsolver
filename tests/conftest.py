"""Shared fixtures: a scripted fake HTTP session and synthetic WAF artefacts."""

import base64
import io
import json
from typing import Callable, List, Tuple

import pytest
import requests

TOKEN_BASE = "https://abc123.def456.eu-central-1.token.awswaf.com/abc123/111/222"
CAPTCHA_BASE = "https://abc123.def456.eu-central-1.captcha.awswaf.com/abc123/111/222"

CAPTCHA_PAGE = f"""<!DOCTYPE html><html><head><title>Human Verification</title>
<script type="text/javascript">
window.awsWafCookieDomainList = [];
window.gokuProps = {{
"key":"AQIDAHjFAKEKEY==",
          "iv":"EkQIdQFAKEIV",
          "context":"Jk+2FAKECONTEXT=="
}};
</script>
<script src="{TOKEN_BASE}/challenge.js"></script>
<script src="{CAPTCHA_BASE}/captcha.js"></script>
</head><body><div id="captcha-container"></div>
<script>CaptchaScript.renderCaptcha(container, async (voucher) => {{}});</script>
</body></html>"""

CHALLENGE_ONLY_PAGE = f"""<html><head><script>window.gokuProps = {{"key":"K","iv":"I","context":"C"}};
</script><script src="{TOKEN_BASE}/challenge.js"></script></head><body></body></html>"""


def jpeg_b64(color=(200, 10, 10), size=100) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def grid_problem(target="clock", n=9, region="eu-central-1") -> dict:
    return {
        "problem_type": "gridcaptcha-5-0.1-0",
        "state": {"iv": "IV", "payload": "PAYLOAD"},
        "key": "PKEY",
        "hmac_tag": "TAG",
        "assets": {"images": json.dumps([jpeg_b64((i * 20, 0, 0)) for i in range(n)]),
                   "target": json.dumps([target]), "res": "320"},
        "localized_assets": {"target0": f"the {target}s"},
        "validity_duration": 60,
        "region": region,
    }


class FakeSession(requests.Session):
    """
    requests.Session whose request() is answered by scripted routes.
    A route is (method, url_prefix, handler(request_kwargs) -> (status, body[, headers])).
    Routes are matched in order; a route with times=1 is consumed after use.
    """

    def __init__(self):
        super().__init__()
        self.routes: List[list] = []
        self.calls: List[Tuple[str, str, dict]] = []

    def add(self, method: str, prefix: str, handler: Callable, times: int = 0):
        self.routes.append([method.upper(), prefix, handler, times])
        return self

    def request(self, method, url, **kwargs):
        self.calls.append((method.upper(), url, kwargs))
        for route in self.routes:
            m, prefix, handler, times = route
            if m == method.upper() and url.startswith(prefix):
                if times == 1:
                    self.routes.remove(route)
                elif times > 1:
                    route[3] = times - 1
                result = handler(kwargs)
                if isinstance(result, Exception):
                    raise result
                status, body, *extra = result
                resp = requests.Response()
                resp.status_code = status
                resp.url = url
                if isinstance(body, (dict, list)):
                    content = json.dumps(body).encode()
                    resp.headers["content-type"] = "application/json"
                elif isinstance(body, bytes):
                    content = body
                else:
                    content = str(body).encode()
                    resp.headers["content-type"] = "text/html"
                if extra:
                    resp.headers.update(extra[0])
                resp.raw = io.BytesIO(content)  # streamed like a real response
                resp.encoding = "utf-8"
                return resp
        raise AssertionError(f"unexpected request {method} {url}")


@pytest.fixture
def fake_session():
    return FakeSession()


PUBLIC_IP = "93.184.216.34"


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    """
    Offline DNS for the SSRF guard: every name resolves to a public address unless the
    test maps it (name -> list of addresses, or an OSError instance to fail resolution).
    """
    from aws_capsolver.waf import http as httpmod
    table = {}

    def resolve(host):
        answer = table.get(host, [PUBLIC_IP])
        if isinstance(answer, Exception):
            raise answer
        return list(answer)

    monkeypatch.setattr(httpmod, "_resolve", resolve)
    return table


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    """
    Fail any TCP connection to a non-loopback address, so protocol/service tests can
    never reach a real site. The model tests (vision/audio) may fetch weights and are
    left alone.
    """
    if request.node.path.name in ("test_vision.py", "test_audio.py"):
        return
    import ipaddress
    import socket

    real_connect = socket.socket.connect

    def guarded(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            host = address[0]
            try:
                local = ipaddress.ip_address(host).is_loopback
            except ValueError:
                local = host == "localhost"
            if not local:
                raise OSError(f"network access disabled in tests: {host}")
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
