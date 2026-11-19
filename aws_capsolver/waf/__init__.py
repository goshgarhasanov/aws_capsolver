"""AWS WAF protocol layer: page parsing, proof-of-work challenge and CAPTCHA exchange."""

from .errors import SolverError
from .parse import WafParams, extract_waf_params, html_is_waf_page
from .urls import captcha_base_url, token_base_url

__all__ = [
    "SolverError",
    "WafParams",
    "extract_waf_params",
    "html_is_waf_page",
    "captcha_base_url",
    "token_base_url",
]
