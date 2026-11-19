import json

import pytest
import requests

from aws_capsolver import vision_bridge
from aws_capsolver.waf import captcha as cap
from aws_capsolver.waf import solver as solvermod
from aws_capsolver.waf.challenge import HASH_A9F, HASH_SHA256, solve_challenge
from aws_capsolver.waf.errors import SolverError

from .conftest import CAPTCHA_BASE, CAPTCHA_PAGE, CHALLENGE_ONLY_PAGE, TOKEN_BASE, grid_problem


def _inputs(challenge_type, difficulty=1):
    return lambda kw: (200, {"challenge": {"input": "uuid:abc:blob", "region": "eu-central-1"},
                             "challenge_type": challenge_type, "difficulty": difficulty})


def test_challenge_bandwidth_multipart(fake_session):
    seen = {}

    def mp_verify(kw):
        seen["files"] = kw["files"]
        return 200, {"token": "TOKEN-BW"}

    fake_session.add("GET", f"{TOKEN_BASE}/inputs", _inputs(HASH_A9F))
    fake_session.add("POST", f"{TOKEN_BASE}/mp_verify", mp_verify)
    assert solve_challenge(fake_session, TOKEN_BASE, "shop.example") == "TOKEN-BW"
    body = json.loads(seen["files"]["challenge"][1])
    assert body["solution"] is None and body["domain"] == "shop.example"
    assert body["client"] == "Browser"
    assert len(seen["files"]["solution_data"][1]) > 1000


def test_challenge_falls_back_on_400(fake_session):
    fake_session.add("GET", f"{TOKEN_BASE}/inputs", _inputs(HASH_A9F))
    fake_session.add("POST", f"{TOKEN_BASE}/mp_verify", lambda kw: (400, {"message": "bad"}))
    got = {}

    def verify(kw):
        got["body"] = kw["json"]
        return 200, {"token": "TOKEN-SCRYPT"}

    fake_session.add("POST", f"{TOKEN_BASE}/verify", verify)
    assert solve_challenge(fake_session, TOKEN_BASE, "d", sleep=lambda s: None) == "TOKEN-SCRYPT"
    assert got["body"]["solution"].isdigit()


def test_challenge_sha256_solution_is_valid(fake_session):
    import hashlib
    fake_session.add("GET", f"{TOKEN_BASE}/inputs", _inputs(HASH_SHA256, 8))
    got = {}

    def verify(kw):
        got["solution"] = kw["json"]["solution"]
        return 200, {"token": "T"}

    fake_session.add("POST", f"{TOKEN_BASE}/verify", verify)
    solve_challenge(fake_session, TOKEN_BASE, "d", processes=1)
    assert hashlib.sha256(f"uuid:abc:blob{got['solution']}".encode()).digest()[0] == 0


def test_challenge_network_error_maps_to_code(fake_session):
    fake_session.add("GET", f"{TOKEN_BASE}/inputs",
                     lambda kw: requests.exceptions.ConnectTimeout("slow"))
    with pytest.raises(SolverError) as err:
        solve_challenge(fake_session, TOKEN_BASE, "d")
    assert err.value.code == "ERROR_TASK_TIMEOUT"
    fake_session.proxies = {"https": "http://p:1"}
    fake_session.routes.clear()
    fake_session.add("GET", f"{TOKEN_BASE}/inputs",
                     lambda kw: requests.exceptions.ProxyError("refused"))
    with pytest.raises(SolverError) as err:
        solve_challenge(fake_session, TOKEN_BASE, "d")
    assert err.value.code == "ERROR_PROXY_CONNECT_REFUSED"


def test_as_list_and_solutions():
    assert cap.as_list('["a","b"]') == ["a", "b"]
    assert cap.as_list(["x"]) == ["x"]
    assert cap.as_list("[broken") == []
    assert cap.as_list(None) == []
    assert cap.toycar_solution(100, 50) == {"x": 5, "y": 3}
    assert cap.normalise_audio_answer("  Hello World ") == "hello world"


class _Calls(list):
    """(n_images, target) per classify_grid call; `.counts` holds the count keyword."""
    counts: list


@pytest.fixture
def fake_vision(monkeypatch):
    calls = _Calls()
    calls.counts = []

    def classify_grid(images, target, count=None):
        calls.append((len(images), target))
        calls.counts.append(count)
        return [4, 1] if len(calls) > 1 else [0]

    monkeypatch.setattr(vision_bridge, "classify_grid", classify_grid)
    return calls


def test_captcha_retry_then_voucher(fake_session, fake_vision):
    verifies = []

    def verify(kw):
        body = json.loads(kw["data"])
        verifies.append(body)
        if len(verifies) == 1:
            return 200, {"success": False, "reason": "wrong", "problem": grid_problem("bed")}
        return 200, {"success": True, "captcha_voucher": "VOUCHER"}

    fake_session.add("GET", f"{CAPTCHA_BASE}/problem", lambda kw: (200, grid_problem()))
    fake_session.add("POST", f"{CAPTCHA_BASE}/verify", verify)
    fake_session.add("POST", f"{TOKEN_BASE}/voucher",
                     lambda kw: (200, {"token": "FINAL"}) if kw["json"]["captcha_voucher"] ==
                     "VOUCHER" and kw["json"]["existing_token"] == "POW" else (400, {}))
    ctx = cap.CaptchaContext(captcha_base=CAPTCHA_BASE, token_base=TOKEN_BASE, domain="d",
                             goku_props={"key": "k", "iv": "i", "context": "c"})
    result = cap.solve_captcha(fake_session, ctx, existing_token="POW")
    assert result.token == "FINAL" and result.rounds == 2
    assert fake_vision == [(9, "clock"), (9, "bed")]
    assert fake_vision.counts == [5, 5]  # from problem_type gridcaptcha-5-...
    assert verifies[0]["client_solution"] == [0] and verifies[1]["client_solution"] == [1, 4]
    assert verifies[0]["goku_props"] == {"key": "k", "iv": "i", "context": "c"}
    assert verifies[0]["metrics"]["solve_time_millis"] > 0
    verify_call = [c for c in fake_session.calls if "/verify" in c[1]][0]
    assert verify_call[1].endswith("/verify?region=eu-central-1")
    assert verify_call[2]["headers"]["content-type"].startswith("text/plain")
    problem_call = fake_session.calls[0]
    assert problem_call[2]["params"] == {"kind": "visual", "domain": "d", "locale": "en-us"}


def test_captcha_gives_up_after_rounds(fake_session, fake_vision):
    fake_session.add("GET", f"{CAPTCHA_BASE}/problem", lambda kw: (200, grid_problem()))
    fake_session.add("POST", f"{CAPTCHA_BASE}/verify",
                     lambda kw: (200, {"success": False, "problem": grid_problem()}))
    ctx = cap.CaptchaContext(CAPTCHA_BASE, TOKEN_BASE, "d", None)
    with pytest.raises(SolverError) as err:
        cap.solve_captcha(fake_session, ctx, max_rounds=3)
    assert err.value.code == "ERROR_CAPTCHA_UNSOLVABLE"


def test_captcha_without_vision_falls_to_audio(fake_session, monkeypatch):
    monkeypatch.setattr(vision_bridge, "_module", False)  # vision package missing
    fake_session.add("GET", f"{CAPTCHA_BASE}/problem", lambda kw: (200, grid_problem()))
    ctx = cap.CaptchaContext(CAPTCHA_BASE, TOKEN_BASE, "d", None)
    with pytest.raises(SolverError) as err:
        cap.solve_captcha(fake_session, ctx)
    assert err.value.code == "ERROR_CAPTCHA_UNSOLVABLE"
    kinds = [c[2]["params"]["kind"] for c in fake_session.calls]
    assert kinds == ["visual", "audio"]


def _patch_session(monkeypatch, session):
    monkeypatch.setattr(solvermod, "make_session", lambda proxy, ua: session.headers.update(
        {"user-agent": ua or "UA-default"}) or session)


def test_solve_website_pow_only(fake_session, monkeypatch):
    _patch_session(monkeypatch, fake_session)
    fake_session.add("GET", "https://shop.example/", lambda kw: (200, CHALLENGE_ONLY_PAGE), times=1)
    fake_session.add("GET", f"{TOKEN_BASE}/inputs", _inputs(HASH_A9F))
    fake_session.add("POST", f"{TOKEN_BASE}/mp_verify", lambda kw: (200, {"token": "TOK"}))
    fake_session.add("GET", "https://shop.example/", lambda kw: (200, "<html>shop</html>"))
    out = solvermod.solve_website(solvermod.SolveRequest("https://shop.example/"),
                                  solvermod.SolverSettings())
    assert out.token == "TOK" and not out.used_captcha and out.user_agent == "UA-default"
    assert fake_session.calls[-1][2]["headers"]["cookie"] == "aws-waf-token=TOK"


def test_solve_website_captcha_page(fake_session, monkeypatch, fake_vision):
    _patch_session(monkeypatch, fake_session)
    fake_session.add("GET", "https://shop.example/", lambda kw: (405, CAPTCHA_PAGE))
    fake_session.add("GET", f"{TOKEN_BASE}/inputs", _inputs(HASH_A9F))
    fake_session.add("POST", f"{TOKEN_BASE}/mp_verify", lambda kw: (200, {"token": "POW"}))
    fake_session.add("GET", f"{CAPTCHA_BASE}/problem", lambda kw: (200, grid_problem()))
    fake_session.add("POST", f"{CAPTCHA_BASE}/verify",
                     lambda kw: (200, {"success": True, "captcha_voucher": "V"}))
    fake_session.add("POST", f"{TOKEN_BASE}/voucher", lambda kw: (200, {"token": "CAPTOK"}))
    out = solvermod.solve_website(solvermod.SolveRequest("https://shop.example/", user_agent="UA"),
                                  solvermod.SolverSettings())
    assert out.token == "CAPTOK" and out.used_captcha and out.voucher == "V"
    verify_body = json.loads([c for c in fake_session.calls if "/verify?" in c[1]][0][2]["data"])
    assert verify_body["goku_props"]["key"] == "AQIDAHjFAKEKEY=="


def test_solve_website_no_waf(fake_session, monkeypatch):
    _patch_session(monkeypatch, fake_session)
    fake_session.add("GET", "https://plain.example/", lambda kw: (200, "<html>hi</html>"))
    with pytest.raises(SolverError) as err:
        solvermod.solve_website(solvermod.SolveRequest("https://plain.example/"),
                                solvermod.SolverSettings())
    assert err.value.code == "ERROR_CAPTCHA_UNSOLVABLE"


def test_solve_website_bad_url():
    with pytest.raises(SolverError) as err:
        solvermod.solve_website(solvermod.SolveRequest("ftp://x"), solvermod.SolverSettings())
    assert err.value.code == "ERROR_INVALID_TASK_DATA"


def test_goku_only_params_still_fetch_page(fake_session, monkeypatch):
    _patch_session(monkeypatch, fake_session)
    fake_session.add("GET", "https://shop.example/", lambda kw: (200, CHALLENGE_ONLY_PAGE))
    fake_session.add("GET", f"{TOKEN_BASE}/inputs", _inputs(HASH_A9F))
    fake_session.add("POST", f"{TOKEN_BASE}/mp_verify", lambda kw: (200, {"token": "TOK"}))
    out = solvermod.solve_website(
        solvermod.SolveRequest("https://shop.example/", aws_key="K2", aws_iv="I2",
                               aws_context="C2"),
        solvermod.SolverSettings(verify_token=False))
    assert out.token == "TOK"


# --- challenge / captcha edge cases ------------------------------------------------------
import time  # noqa: E402

from aws_capsolver.waf.parse import extract_waf_params  # noqa: E402


def test_challenge_accepts_numeric_string_difficulty(fake_session):
    fake_session.add("GET", f"{TOKEN_BASE}/inputs", lambda kw: (200, {
        "challenge": {"input": "i"}, "challenge_type": HASH_A9F, "difficulty": "1"}))
    fake_session.add("POST", f"{TOKEN_BASE}/mp_verify", lambda kw: (200, {"token": "T"}))
    assert solve_challenge(fake_session, TOKEN_BASE, "d") == "T"


@pytest.mark.parametrize("inputs", [
    {"challenge": {"input": "i"}, "challenge_type": 5, "difficulty": 1},
    {"challenge": {"input": "i"}, "challenge_type": HASH_A9F, "difficulty": True},
    {"challenge": "i", "challenge_type": HASH_A9F, "difficulty": 1},
    {"challenge": {"input": ""}, "challenge_type": HASH_A9F, "difficulty": 1},
])
def test_challenge_rejects_malformed_inputs(fake_session, inputs):
    fake_session.add("GET", f"{TOKEN_BASE}/inputs", lambda kw: (200, inputs))
    with pytest.raises(SolverError) as err:
        solve_challenge(fake_session, TOKEN_BASE, "d")
    assert err.value.code == "ERROR_CAPTCHA_UNSOLVABLE"


def test_challenge_all_candidates_rejected(fake_session):
    fake_session.add("GET", f"{TOKEN_BASE}/inputs", _inputs(HASH_A9F))
    fake_session.add("POST", f"{TOKEN_BASE}/mp_verify", lambda kw: (400, {}))
    fake_session.add("POST", f"{TOKEN_BASE}/verify", lambda kw: (400, {}))
    with pytest.raises(SolverError) as err:
        solve_challenge(fake_session, TOKEN_BASE, "d", sleep=lambda s: None, processes=1)
    assert err.value.code == "ERROR_CAPTCHA_UNSOLVABLE"
    assert len([c for c in fake_session.calls if c[1].endswith("/inputs")]) == 3


def test_challenge_non_string_token(fake_session):
    fake_session.add("GET", f"{TOKEN_BASE}/inputs", _inputs(HASH_A9F))
    fake_session.add("POST", f"{TOKEN_BASE}/mp_verify", lambda kw: (200, {"token": 123}))
    with pytest.raises(SolverError):
        solve_challenge(fake_session, TOKEN_BASE, "d")


def test_challenge_deadline(fake_session):
    with pytest.raises(SolverError) as err:
        solve_challenge(fake_session, TOKEN_BASE, "d", deadline=time.monotonic() - 1)
    assert err.value.code == "ERROR_TASK_TIMEOUT"
    assert fake_session.calls == []


def test_grid_tile_count():
    assert cap.grid_tile_count({"problem_type": "gridcaptcha-5-0.1-0"}) == 5
    assert cap.grid_tile_count({"problem_type": "gridcaptcha-x"}) is None
    assert cap.grid_tile_count({"problem_type": "toycarcity-1"}) is None
    assert cap.grid_tile_count({}) is None


def _ctx():
    return cap.CaptchaContext(CAPTCHA_BASE, TOKEN_BASE, "d", None)


def test_bad_base64_grid_image_falls_to_audio(fake_session, fake_vision):
    bad = grid_problem()
    bad["assets"]["images"] = json.dumps(["%%%not-base64%%%"] * 9)
    fake_session.add("GET", f"{CAPTCHA_BASE}/problem", lambda kw: (200, bad))
    with pytest.raises(SolverError) as err:
        cap.solve_captcha(fake_session, _ctx())
    assert err.value.code == "ERROR_CAPTCHA_UNSOLVABLE"
    assert [c[2]["params"]["kind"] for c in fake_session.calls] == ["visual", "audio"]
    assert fake_vision == []  # never reached the model


@pytest.mark.parametrize("problem", [
    {"problem_type": "gridcaptcha-5", "state": "s", "assets": "not-a-dict"},
    {"problem_type": "toycarcity-1", "state": "s", "assets": {"image": 5}},
    {"problem_type": "noisyneighbour-1", "state": "s", "assets": {"audio": ["x"]}},
    {"problem_type": "brandnew-1", "state": "s", "assets": {}},
])
def test_malformed_problems_are_unsolvable(problem):
    with pytest.raises(SolverError) as err:
        cap.solve_problem(problem)
    assert err.value.code == "ERROR_CAPTCHA_UNSOLVABLE"


def test_vision_errors_become_unsolvable(monkeypatch):
    from aws_capsolver.vision.errors import InvalidInput

    def raise_invalid(images, target, count=None):
        raise InvalidInput("9 tiles expected")

    monkeypatch.setattr(vision_bridge, "classify_grid", raise_invalid)
    with pytest.raises(SolverError) as err:
        cap.solve_problem(grid_problem())
    assert err.value.code == "ERROR_CAPTCHA_UNSOLVABLE"
    monkeypatch.setattr(vision_bridge, "classify_grid", lambda i, t, count=None: ["a"])
    with pytest.raises(SolverError):
        cap.solve_problem(grid_problem())
    monkeypatch.setattr(vision_bridge, "classify_grid", lambda i, t, count=None: [])
    with pytest.raises(SolverError):
        cap.solve_problem(grid_problem())


def test_captcha_success_without_voucher_or_problem(fake_session, fake_vision):
    fake_session.add("GET", f"{CAPTCHA_BASE}/problem", lambda kw: (200, grid_problem()))
    fake_session.add("POST", f"{CAPTCHA_BASE}/verify",
                     lambda kw: (200, {"success": True, "problem": "garbage",
                                       "captcha_voucher": 17}))
    with pytest.raises(SolverError) as err:
        cap.solve_captcha(fake_session, _ctx(), kinds=("visual",))
    assert "no voucher and no next problem" in err.value.description


def test_captcha_non_string_region_is_ignored(fake_session, fake_vision):
    problem = grid_problem()
    problem["region"] = {"bad": 1}
    fake_session.add("GET", f"{CAPTCHA_BASE}/problem", lambda kw: (200, problem))
    fake_session.add("POST", f"{CAPTCHA_BASE}/verify",
                     lambda kw: (200, {"success": True, "captcha_voucher": "V"}))
    result = cap.solve_captcha(fake_session, _ctx(), redeem=False)
    assert result.voucher == "V" and result.token is None
    assert [c[1] for c in fake_session.calls][-1] == f"{CAPTCHA_BASE}/verify"


def test_captcha_more_problems_than_rounds(fake_session, fake_vision):
    fake_session.add("GET", f"{CAPTCHA_BASE}/problem", lambda kw: (200, grid_problem()))
    fake_session.add("POST", f"{CAPTCHA_BASE}/verify",
                     lambda kw: (200, {"success": True, "problem": grid_problem()}))
    with pytest.raises(SolverError) as err:
        cap.solve_captcha(fake_session, _ctx(), max_rounds=2)
    assert "more problems required" in err.value.description
    assert len([c for c in fake_session.calls if "/verify" in c[1]]) == 2
    assert len([c for c in fake_session.calls if "/problem" in c[1]]) == 1  # no audio retry


def test_captcha_deadline_and_cancel(fake_session, fake_vision):
    import threading
    with pytest.raises(SolverError) as err:
        cap.solve_captcha(fake_session, _ctx(), deadline=time.monotonic() - 1)
    assert err.value.code == "ERROR_TASK_TIMEOUT"
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(SolverError) as err:
        cap.solve_captcha(fake_session, _ctx(), cancel=cancel)
    assert err.value.code == "ERROR_SERVICE_UNAVAILABLE"
    assert fake_session.calls == []


def test_voucher_network_failure_is_reported(fake_session, fake_vision):
    fake_session.add("GET", f"{CAPTCHA_BASE}/problem", lambda kw: (200, grid_problem()))
    fake_session.add("POST", f"{CAPTCHA_BASE}/verify",
                     lambda kw: (200, {"success": True, "captcha_voucher": "V"}))
    fake_session.add("POST", f"{TOKEN_BASE}/voucher",
                     lambda kw: requests.exceptions.ConnectionError("down"))
    with pytest.raises(SolverError) as err:
        cap.solve_captcha(fake_session, _ctx())
    assert err.value.code == "ERROR_SITE_UNREACHABLE"


def test_extract_escaped_script_urls_and_odd_goku():
    escaped = TOKEN_BASE.replace("/", "\\/")
    html = ('<script>var cfg = {"src": "' + escaped + '\\/challenge.js"};'
            'window.gokuProps = {"key": "K", "iv": 5, "context": null};</script>')
    params = extract_waf_params(html)
    assert params.challenge_js == f"{TOKEN_BASE}/challenge.js"
    assert params.key == "K" and params.iv is None and params.goku_props is None


def test_solve_website_captcha_endpoint_from_refreshed_page(fake_session, monkeypatch,
                                                           fake_vision):
    """PoW token rejected by the page -> captcha via the endpoint named on that page."""
    _patch_session(monkeypatch, fake_session)
    fake_session.add("GET", "https://shop.example/", lambda kw: (200, CHALLENGE_ONLY_PAGE),
                     times=1)
    fake_session.add("GET", f"{TOKEN_BASE}/inputs", _inputs(HASH_A9F))
    fake_session.add("POST", f"{TOKEN_BASE}/mp_verify", lambda kw: (200, {"token": "POW"}))
    fake_session.add("GET", "https://shop.example/", lambda kw: (405, CAPTCHA_PAGE))
    fake_session.add("GET", f"{CAPTCHA_BASE}/problem", lambda kw: (200, grid_problem()))
    fake_session.add("POST", f"{CAPTCHA_BASE}/verify",
                     lambda kw: (200, {"success": True, "captcha_voucher": "V"}))
    fake_session.add("POST", f"{TOKEN_BASE}/voucher", lambda kw: (200, {"token": "CAPTOK"}))
    out = solvermod.solve_website(solvermod.SolveRequest("https://shop.example/"),
                                  solvermod.SolverSettings())
    assert out.token == "CAPTOK" and out.used_captcha
    verify_body = json.loads([c for c in fake_session.calls if "/verify?" in c[1]][0][2]["data"])
    assert verify_body["goku_props"]["key"] == "AQIDAHjFAKEKEY=="
