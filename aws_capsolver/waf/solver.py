"""End-to-end AWS WAF solving for one website: parameters -> PoW -> captcha -> token."""

import logging
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests

from .captcha import CaptchaContext, solve_captcha
from .challenge import solve_challenge
from .errors import ERROR_CAPTCHA_UNSOLVABLE, ERROR_INVALID_TASK_DATA, SolverError
from .http import make_session, map_network_error
from .parse import WafParams, extract_waf_params, html_is_waf_page
from .urls import captcha_base_url, token_base_url

logger = logging.getLogger(__name__)

WAF_COOKIE = "aws-waf-token"
BLOCK_STATUSES = (202, 403, 405)


@dataclass
class SolveRequest:
    website_url: str
    proxy_url: Optional[str] = None
    user_agent: Optional[str] = None
    aws_key: Optional[str] = None
    aws_iv: Optional[str] = None
    aws_context: Optional[str] = None
    challenge_js: Optional[str] = None
    api_js: Optional[str] = None
    problem_url: Optional[str] = None
    api_key: Optional[str] = None
    locale: str = "en-us"


@dataclass
class SolveOutcome:
    token: Optional[str]
    user_agent: str
    captcha_rounds: int = 0
    used_captcha: bool = False
    voucher: Optional[str] = None


@dataclass
class SolverSettings:
    http_timeout: float = 20.0
    pow_processes: Optional[int] = None
    captcha_rounds: int = 5
    verify_token: bool = True


def _fetch_page(session: requests.Session, url: str, timeout: float, cookie: Optional[str] = None):
    headers = {"accept": "text/html,application/xhtml+xml,*/*;q=0.8"}
    if cookie:
        headers["cookie"] = f"{WAF_COOKIE}={cookie}"
    try:
        resp = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    except requests.RequestException as exc:
        raise map_network_error(exc, "website", bool(session.proxies)) from exc
    if resp.status_code == 407:
        raise SolverError("ERROR_PROXY_CONNECT_REFUSED", "proxy authentication required")
    return resp.status_code, resp.text


def _merge_params(req: SolveRequest, page: Optional[WafParams]) -> WafParams:
    params = page or WafParams()
    params.key = req.aws_key or params.key
    params.iv = req.aws_iv or params.iv
    params.context = req.aws_context or params.context
    params.challenge_js = req.challenge_js or params.challenge_js
    params.jsapi_js = req.api_js or params.jsapi_js
    params.api_key = req.api_key or params.api_key
    return params


def solve_website(req: SolveRequest, settings: SolverSettings,
                  deadline: Optional[float] = None) -> SolveOutcome:
    parsed = urlparse(req.website_url or "")
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise SolverError(ERROR_INVALID_TASK_DATA, "websiteURL must be an absolute http(s) URL")
    domain = parsed.hostname
    session = make_session(req.proxy_url, req.user_agent)
    user_agent = session.headers["user-agent"]
    t = settings.http_timeout

    # gokuProps alone are not enough: the script URLs name the WAF endpoints
    have_params = bool(req.challenge_js or req.problem_url)
    page_status, page_params = None, None
    if not have_params:
        page_status, html = _fetch_page(session, req.website_url, t)
        page_params = extract_waf_params(html)
        logger.info("page %s -> HTTP %s waf=%s captcha=%s", domain, page_status,
                    html_is_waf_page(html), page_params.captcha_rendered)
        if not page_params.has_anything:
            raise SolverError(ERROR_CAPTCHA_UNSOLVABLE,
                              f"no AWS WAF challenge found on the page (HTTP {page_status})")
    params = _merge_params(req, page_params)

    script = params.challenge_js or params.captcha_js or req.problem_url or params.jsapi_js
    token_base = token_base_url(params.challenge_js or "") or token_base_url(script or "")
    captcha_base = captcha_base_url(req.problem_url or "") or captcha_base_url(
        params.captcha_js or "") or captcha_base_url(params.challenge_js or "")
    if not token_base and not captcha_base:
        raise SolverError(ERROR_INVALID_TASK_DATA, "could not derive the awswaf endpoints; "
                          "pass awsChallengeJS or awsProblemUrl")

    captcha_needed = bool(req.problem_url) or bool(params.jsapi_js and params.api_key) or (
        page_status in BLOCK_STATUSES and params.captcha_rendered)

    token: Optional[str] = None
    if token_base:
        token = solve_challenge(session, token_base, domain, timeout=t,
                                processes=settings.pow_processes, deadline=deadline)

    if not captcha_needed and token and settings.verify_token and page_status is not None:
        status, html = _fetch_page(session, req.website_url, t, cookie=token)
        still_blocked = status in BLOCK_STATUSES and html_is_waf_page(html)
        logger.info("token check %s -> HTTP %s blocked=%s", domain, status, still_blocked)
        if still_blocked:
            captcha_needed = True
            refreshed = extract_waf_params(html)
            if refreshed.goku_props:
                params.key, params.iv, params.context = (refreshed.key, refreshed.iv,
                                                         refreshed.context)

    if not captcha_needed:
        if not token:
            raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, "no token produced")
        return SolveOutcome(token=token, user_agent=user_agent)

    if not captcha_base:
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, "captcha required but no captcha endpoint")
    ctx = CaptchaContext(captcha_base=captcha_base, token_base=token_base, domain=domain,
                         goku_props=params.goku_props, api_key=params.api_key, locale=req.locale)
    result = solve_captcha(session, ctx, existing_token=token, max_rounds=settings.captcha_rounds,
                           timeout=t, deadline=deadline, redeem=bool(token_base))
    final = result.token or token
    if not final and not result.voucher:
        raise SolverError(ERROR_CAPTCHA_UNSOLVABLE, "captcha voucher could not be redeemed")
    return SolveOutcome(token=final, user_agent=user_agent, captcha_rounds=result.rounds,
                        used_captcha=True, voucher=result.voucher)


def solve_website_simple(url: str, **kwargs) -> SolveOutcome:
    """Convenience wrapper used by the CLI."""
    return solve_website(SolveRequest(website_url=url, **kwargs), SolverSettings(),
                         deadline=time.monotonic() + 120)
