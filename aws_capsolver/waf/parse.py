"""Extraction of the AWS WAF parameters from a protected page's HTML."""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

_GOKU_OBJECT_RE = re.compile(r"window\.gokuProps\s*=\s*(\{.*?\})\s*;?\s*</script>", re.S | re.I)
_GOKU_OBJECT_LOOSE_RE = re.compile(r"gokuProps\s*=\s*(\{.*?\})", re.S)
_FIELD_RE = {
    name: re.compile(r"[\"']?%s[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']" % name)
    for name in ("key", "iv", "context")
}
_CHALLENGE_JS_RE = re.compile(r"https://[\w.\-]+\.token\.awswaf\.com/[\w/\-]+/challenge\.js", re.I)
_CAPTCHA_JS_RE = re.compile(r"https://[\w.\-]+\.captcha\.awswaf\.com/[\w/\-]+/captcha\.js", re.I)
_JSAPI_RE = re.compile(r"https://[\w.\-]+\.(?:captcha-sdk|jsapi)\.awswaf\.com/[\w/\-]+/jsapi\.js", re.I)
_API_KEY_RE = re.compile(r"[\"']?apiKey[\"']?\s*[:=]\s*[\"']([^\"']{20,})[\"']")

_WAF_MARKERS = ("gokuProps", "token.awswaf.com", "captcha.awswaf.com", "AwsWafIntegration",
                "CaptchaScript.renderCaptcha", "AwsWafCaptcha")


@dataclass
class WafParams:
    """Everything needed to talk to the WAF on behalf of one page."""

    key: Optional[str] = None
    iv: Optional[str] = None
    context: Optional[str] = None
    challenge_js: Optional[str] = None
    captcha_js: Optional[str] = None
    jsapi_js: Optional[str] = None
    api_key: Optional[str] = None
    captcha_rendered: bool = False
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def goku_props(self) -> Optional[Dict[str, str]]:
        if self.key and self.iv and self.context:
            return {"key": self.key, "iv": self.iv, "context": self.context}
        return None

    @property
    def has_anything(self) -> bool:
        return bool(self.goku_props or self.challenge_js or self.captcha_js or self.jsapi_js)


def html_is_waf_page(html: str) -> bool:
    """True when the HTML looks like an AWS WAF challenge or captcha interstitial."""
    return bool(html) and any(m in html for m in _WAF_MARKERS)


def _parse_goku(html: str) -> Dict[str, str]:
    props: Dict[str, str] = {}
    for rx in (_GOKU_OBJECT_RE, _GOKU_OBJECT_LOOSE_RE):
        match = rx.search(html)
        if not match:
            continue
        block = match.group(1)
        try:
            obj = json.loads(block)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            props = {k: v for k, v in obj.items()
                     if k in ("key", "iv", "context") and isinstance(v, str) and v}
        else:
            # not strict JSON (single quotes, unquoted keys) - read field by field
            for name, frx in _FIELD_RE.items():
                found = frx.search(block)
                if found:
                    props[name] = found.group(1)
        if props:
            break
    return props


def extract_waf_params(html: str) -> WafParams:
    """Parse gokuProps, script URLs and the jsapi apiKey out of a page."""
    params = WafParams()
    if not html:
        return params
    goku = _parse_goku(html)
    params.key, params.iv, params.context = goku.get("key"), goku.get("iv"), goku.get("context")
    scan = html.replace("\\/", "/")  # URLs inside JSON strings are often written https:\/\/
    for attr, rx in (("challenge_js", _CHALLENGE_JS_RE), ("captcha_js", _CAPTCHA_JS_RE),
                     ("jsapi_js", _JSAPI_RE)):
        found = rx.search(scan)
        if found:
            setattr(params, attr, found.group(0))
    if params.jsapi_js:
        found = _API_KEY_RE.search(html)
        if found:
            params.api_key = found.group(1)
    params.captcha_rendered = "renderCaptcha" in html or bool(params.captcha_js)
    return params
