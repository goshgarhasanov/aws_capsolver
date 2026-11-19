"""HTTP plumbing: proxy parsing, error mapping, bounded reads, SSRF guard, redirects."""

import logging
import time

import pytest
import requests

from aws_capsolver.waf import solver as solvermod
from aws_capsolver.waf.errors import SolverError
from aws_capsolver.waf.http import (RedactingFilter, check_public_host, is_public_address,
                                    map_network_error, parse_proxy, redact, remaining_timeout,
                                    request_json)

from .conftest import CHALLENGE_ONLY_PAGE, TOKEN_BASE


# --- proxy parsing ---------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("http://u:p@h:1", "http://u:p@h:1"),
    ("socks5h://u:p@h.example:1080", "socks5h://u:p@h.example:1080"),
    ("http://u:p@[2001:db8::1]:3128", "http://u:p@[2001:db8::1]:3128"),
    ("  h:3128  ", "http://h:3128"),
    ("https:h:443", "https://h:443"),
    ("socks4:h:1080", "socks4://h:1080"),
    ("h:3128:user@mail:p@ss:w0rd", "http://user%40mail:p%40ss%3Aw0rd@h:3128"),
    ("h:3128:u:p/#?", "http://u:p%2F%23%3F@h:3128"),
    ({"proxyType": "SOCKS5", "proxyAddress": " 1.2.3.4 ", "proxyPort": "1080"},
     "socks5://1.2.3.4:1080"),
    ({"proxyAddress": "h", "proxyPort": 8, "proxyLogin": "u@x", "proxyPassword": "a:b"},
     "http://u%40x:a%3Ab@h:8"),
    ({"proxyAddress": "h", "proxyPort": 8, "proxyLogin": "u"}, "http://u:@h:8"),
    (None, None),
    ("", None),
    ("   ", None),
    ({}, None),
])
def test_parse_proxy_matrix(raw, expected):
    assert parse_proxy(raw) == expected


@pytest.mark.parametrize("raw", [
    "h:0", "h:65536", "h:-1", "h:80a", ":80", "h:1:u", "x",
    "http://h:abc", "http://h:99999", "http://h", "ftp://h:21", "h o:80", "h/x:80",
    "h:80:u p:pw", 12345, ["h:1"],
    {"proxyAddress": "h"}, {"proxyAddress": "h", "proxyPort": "x"},
    {"proxyAddress": "h", "proxyPort": 0}, {"proxyAddress": "h/evil", "proxyPort": 1},
    {"proxyType": "ftp", "proxyAddress": "h", "proxyPort": 1},
])
def test_parse_proxy_rejects_matrix(raw):
    with pytest.raises(SolverError) as err:
        parse_proxy(raw)
    assert err.value.code == "ERROR_INVALID_TASK_DATA"


# --- redaction -------------------------------------------------------------------------
def test_redact_masks_passwords():
    text = "proxy http://alice:s3cr3t@h:1 and socks5://bob:p%40ss@x:2 failed"
    out = redact(text)
    assert "s3cr3t" not in out and "p%40ss" not in out
    assert "http://alice:***@h:1" in out and "socks5://bob:***@x:2" in out
    assert redact("https://example.com/a@b") == "https://example.com/a@b"


def test_redacting_filter_cleans_log_records(caplog):
    log = logging.getLogger("aws_capsolver.test_redaction")
    caplog.handler.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO):
        log.info("using %s", "http://u:topsecret@h:1")
    assert "topsecret" not in caplog.text and "u:***@h:1" in caplog.text


# --- error mapping ---------------------------------------------------------------------
@pytest.mark.parametrize("exc,via_proxy,code", [
    (requests.exceptions.ProxyError("x"), True, "ERROR_PROXY_CONNECT_REFUSED"),
    (requests.exceptions.InvalidProxyURL("x"), True, "ERROR_INVALID_TASK_DATA"),
    (requests.exceptions.ConnectTimeout("x"), True, "ERROR_PROXY_CONNECT_REFUSED"),
    (requests.exceptions.ConnectTimeout("x"), False, "ERROR_TASK_TIMEOUT"),
    (requests.exceptions.ReadTimeout("x"), True, "ERROR_TASK_TIMEOUT"),
    (requests.exceptions.SSLError("x"), False, "ERROR_SITE_UNREACHABLE"),
    (requests.exceptions.ConnectionError("x"), False, "ERROR_SITE_UNREACHABLE"),
    (requests.exceptions.ConnectionError("x"), True, "ERROR_PROXY_CONNECT_REFUSED"),
    (requests.exceptions.InvalidURL("x"), False, "ERROR_INVALID_TASK_DATA"),
    (requests.exceptions.MissingSchema("x"), False, "ERROR_INVALID_TASK_DATA"),
    (requests.exceptions.TooManyRedirects("x"), False, "ERROR_SITE_UNREACHABLE"),
    (requests.exceptions.ChunkedEncodingError("x"), False, "ERROR_SITE_UNREACHABLE"),
])
def test_map_network_error(exc, via_proxy, code):
    assert map_network_error(exc, "what", via_proxy).code == code


@pytest.mark.parametrize("status,body,proxied,code,http_status", [
    (400, {"m": 1}, False, "ERROR_CAPTCHA_UNSOLVABLE", 400),
    (403, {}, False, "ERROR_CAPTCHA_UNSOLVABLE", 403),
    (403, {}, True, "ERROR_PROXY_BANNED", 403),
    (429, {}, True, "ERROR_PROXY_BANNED", 429),
    (407, {}, True, "ERROR_PROXY_CONNECT_REFUSED", 407),
    (502, {}, False, "ERROR_SITE_UNREACHABLE", 502),
    (200, "not json", False, "ERROR_SITE_UNREACHABLE", None),
    (200, ["a", "list"], False, "ERROR_SITE_UNREACHABLE", None),
])
def test_request_json_status_mapping(fake_session, status, body, proxied, code, http_status):
    if proxied:
        fake_session.proxies = {"https": "http://p:1"}
    fake_session.add("GET", "https://x.example/", lambda kw: (status, body))
    with pytest.raises(SolverError) as err:
        request_json(fake_session, "GET", "https://x.example/", "probe", 5)
    assert err.value.code == code and err.value.http_status == http_status
    assert err.value.code.startswith("ERROR_")


def test_request_json_size_limit(fake_session):
    fake_session.add("GET", "https://x.example/big", lambda kw: (200, b"{" + b" " * 5000 + b"}"))
    with pytest.raises(SolverError) as err:
        request_json(fake_session, "GET", "https://x.example/big", "big", 5, max_bytes=1000)
    assert err.value.code == "ERROR_SITE_UNREACHABLE" and "too large" in err.value.description
    fake_session.add("GET", "https://x.example/declared",
                     lambda kw: (200, b"{}", {"content-length": "999999999"}))
    with pytest.raises(SolverError):
        request_json(fake_session, "GET", "https://x.example/declared", "decl", 5)


def test_request_timeout_is_bounded_by_deadline(fake_session):
    fake_session.add("GET", "https://x.example/", lambda kw: (200, {"t": kw["timeout"]}))
    got = request_json(fake_session, "GET", "https://x.example/", "t", 20,
                       deadline=time.monotonic() + 3)
    assert 0.5 <= got["t"] <= 3
    with pytest.raises(SolverError) as err:
        request_json(fake_session, "GET", "https://x.example/", "t", 20,
                     deadline=time.monotonic() - 1)
    assert err.value.code == "ERROR_TASK_TIMEOUT"
    assert remaining_timeout(7, None) == 7


# --- SSRF guard ------------------------------------------------------------------------
@pytest.mark.parametrize("address,public", [
    ("8.8.8.8", True), ("2606:4700:4700::1111", True),
    ("127.0.0.1", False), ("10.1.2.3", False), ("172.16.0.1", False),
    ("192.168.1.91", False), ("169.254.169.254", False), ("0.0.0.0", False),
    ("100.64.0.1", False), ("::1", False), ("fe80::1", False), ("fc00::1", False),
    ("::ffff:127.0.0.1", False), ("::ffff:10.0.0.1", False), ("224.0.0.1", False),
    ("not-an-ip", False),
])
def test_is_public_address(address, public):
    assert is_public_address(address) is public


def test_check_public_host(fake_dns):
    check_public_host("shop.example")  # resolves to a public address
    fake_dns["localhost"] = ["127.0.0.1"]
    fake_dns["mixed.example"] = ["93.184.216.34", "10.0.0.5"]
    fake_dns["nx.example"] = OSError("nxdomain")
    for host in ("127.0.0.1", "[::1]", "localhost", "mixed.example"):
        with pytest.raises(SolverError) as err:
            check_public_host(host)
        assert err.value.code == "ERROR_INVALID_TASK_DATA"
        assert "AWSCS_ALLOW_PRIVATE" in err.value.description
    with pytest.raises(SolverError) as err:
        check_public_host("nx.example")
    assert err.value.code == "ERROR_SITE_UNREACHABLE"
    check_public_host("nx.example", resolve_failure_ok=True)  # the proxy resolves it
    with pytest.raises(SolverError):
        check_public_host("")


def test_solve_website_blocks_private_targets_before_any_request(fake_session, monkeypatch,
                                                                  fake_dns):
    monkeypatch.setattr(solvermod, "make_session", lambda proxy, ua: fake_session)
    fake_dns["intranet.example"] = ["192.168.0.10"]
    for url in ("http://127.0.0.1:8790/", "http://169.254.169.254/latest/meta-data/",
                "https://intranet.example/", "http://[::1]/"):
        with pytest.raises(SolverError) as err:
            solvermod.solve_website(solvermod.SolveRequest(url), solvermod.SolverSettings())
        assert err.value.code == "ERROR_INVALID_TASK_DATA", url
    with pytest.raises(SolverError) as err:  # a proxy into the private network
        solvermod.solve_website(solvermod.SolveRequest("https://shop.example/",
                                                       proxy_url="http://10.0.0.1:3128"),
                                solvermod.SolverSettings())
    assert err.value.code == "ERROR_INVALID_TASK_DATA"
    assert fake_session.calls == []


def test_allow_private_disables_the_guard(fake_session, monkeypatch):
    monkeypatch.setattr(solvermod, "make_session", lambda proxy, ua: fake_session)
    fake_session.add("GET", "http://127.0.0.1:9/", lambda kw: (200, "<html>plain</html>"))
    with pytest.raises(SolverError) as err:
        solvermod.solve_website(solvermod.SolveRequest("http://127.0.0.1:9/"),
                                solvermod.SolverSettings(allow_private=True))
    assert "no AWS WAF challenge" in err.value.description


def test_redirect_into_private_network_is_blocked(fake_session, monkeypatch):
    monkeypatch.setattr(solvermod, "make_session", lambda proxy, ua: fake_session)
    fake_session.add("GET", "https://shop.example/",
                     lambda kw: (302, "", {"location": "http://127.0.0.1/admin"}))
    with pytest.raises(SolverError) as err:
        solvermod.solve_website(solvermod.SolveRequest("https://shop.example/"),
                                solvermod.SolverSettings())
    assert err.value.code == "ERROR_INVALID_TASK_DATA"
    assert [c[1] for c in fake_session.calls] == ["https://shop.example/"]
    assert fake_session.calls[0][2]["allow_redirects"] is False


def test_public_redirects_are_followed_and_bounded(fake_session, monkeypatch):
    monkeypatch.setattr(solvermod, "make_session", lambda proxy, ua: fake_session)
    fake_session.add("GET", "https://shop.example/start",
                     lambda kw: (301, "", {"location": "/landing"}))
    fake_session.add("GET", "https://shop.example/landing", lambda kw: (200, CHALLENGE_ONLY_PAGE))
    fake_session.add("GET", f"{TOKEN_BASE}/inputs", lambda kw: (200, {
        "challenge": {"input": "i"}, "challenge_type": "unknown", "difficulty": 1}))
    fake_session.add("POST", f"{TOKEN_BASE}/mp_verify", lambda kw: (200, {"token": "TOK"}))
    out = solvermod.solve_website(solvermod.SolveRequest("https://shop.example/start"),
                                  solvermod.SolverSettings(verify_token=False))
    assert out.token == "TOK"
    fake_session.routes.clear()
    fake_session.calls.clear()
    fake_session.add("GET", "https://loop.example/", lambda kw: (302, "", {"location": "/"}))
    with pytest.raises(SolverError) as err:
        solvermod.solve_website(solvermod.SolveRequest("https://loop.example/"),
                                solvermod.SolverSettings())
    assert "too many redirects" in err.value.description
    assert len(fake_session.calls) == solvermod.MAX_REDIRECTS + 1


def test_token_check_failure_keeps_the_token(fake_session, monkeypatch):
    monkeypatch.setattr(solvermod, "make_session", lambda proxy, ua: fake_session)
    fake_session.add("GET", "https://shop.example/", lambda kw: (200, CHALLENGE_ONLY_PAGE),
                     times=1)
    fake_session.add("GET", f"{TOKEN_BASE}/inputs", lambda kw: (200, {
        "challenge": {"input": "i"}, "challenge_type": "unknown", "difficulty": 1}))
    fake_session.add("POST", f"{TOKEN_BASE}/mp_verify", lambda kw: (200, {"token": "TOK"}))
    fake_session.add("GET", "https://shop.example/",
                     lambda kw: requests.exceptions.ReadTimeout("slow"))
    out = solvermod.solve_website(solvermod.SolveRequest("https://shop.example/"),
                                  solvermod.SolverSettings())
    assert out.token == "TOK"


def test_session_is_closed(fake_session, monkeypatch):
    closed = []
    fake_session.close = lambda: closed.append(True)
    monkeypatch.setattr(solvermod, "make_session", lambda proxy, ua: fake_session)
    fake_session.add("GET", "https://plain.example/", lambda kw: (200, "<html></html>"))
    with pytest.raises(SolverError):
        solvermod.solve_website(solvermod.SolveRequest("https://plain.example/"),
                                solvermod.SolverSettings())
    assert closed == [True]
