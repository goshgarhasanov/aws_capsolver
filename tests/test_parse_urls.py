from aws_capsolver.waf.parse import extract_waf_params, html_is_waf_page
from aws_capsolver.waf.urls import captcha_base_url, token_base_url

from .conftest import CAPTCHA_BASE, CAPTCHA_PAGE, CHALLENGE_ONLY_PAGE, TOKEN_BASE


def test_url_derivation_both_ways():
    assert token_base_url(f"{CAPTCHA_BASE}/captcha.js") == TOKEN_BASE
    assert token_base_url(f"{TOKEN_BASE}/challenge.js") == TOKEN_BASE
    assert captcha_base_url(f"{TOKEN_BASE}/challenge.js") == CAPTCHA_BASE
    assert captcha_base_url(f"{CAPTCHA_BASE}/problem") == CAPTCHA_BASE
    assert captcha_base_url(f"{CAPTCHA_BASE}/problem?kind=visual") == CAPTCHA_BASE


def test_url_derivation_rejects_foreign_hosts():
    assert token_base_url("https://evil.example.com/a/b/challenge.js") is None
    assert captcha_base_url("") is None
    assert token_base_url("https://x.awswaf.com.evil.io/a/challenge.js") is None


def test_extract_goku_and_scripts():
    params = extract_waf_params(CAPTCHA_PAGE)
    assert params.goku_props == {"key": "AQIDAHjFAKEKEY==", "iv": "EkQIdQFAKEIV",
                                 "context": "Jk+2FAKECONTEXT=="}
    assert params.challenge_js == f"{TOKEN_BASE}/challenge.js"
    assert params.captcha_js == f"{CAPTCHA_BASE}/captcha.js"
    assert params.captcha_rendered
    assert html_is_waf_page(CAPTCHA_PAGE)


def test_extract_non_json_goku():
    html = "<script>window.gokuProps = {key: 'k1', iv: 'i1', context: 'c1',};</script>"
    assert extract_waf_params(html).goku_props == {"key": "k1", "iv": "i1", "context": "c1"}


def test_extract_challenge_only_and_jsapi():
    params = extract_waf_params(CHALLENGE_ONLY_PAGE)
    assert params.captcha_js is None and not params.captcha_rendered
    html = ('<script src="https://abc.edge.sdk.awswaf.com/x/jsapi.js"></script>'
            '<script src="https://a1.b2.us-east-1.captcha-sdk.awswaf.com/a1/jsapi.js"></script>'
            '<script>AwsWafCaptcha.renderCaptcha(el, {apiKey: "' + "A" * 40 + '"})</script>')
    params = extract_waf_params(html)
    assert params.jsapi_js.endswith("/jsapi.js")
    assert params.api_key == "A" * 40


def test_plain_page_has_nothing():
    params = extract_waf_params("<html><body>hello</body></html>")
    assert not params.has_anything
    assert not html_is_waf_page("<html></html>")
