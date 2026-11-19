"""Command line: `serve` runs the API, `solve` solves one URL and prints the token."""

import argparse
import json
import logging
import os
import sys
import time


def _logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="aws_capsolver")
    parser.add_argument("--log-level", default=os.environ.get("AWSCS_LOG_LEVEL", "INFO"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="run the CapSolver-compatible API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8790)
    serve.add_argument("--open", action="store_true", help="accept any clientKey (dev only)")
    solve = sub.add_parser("solve", help="solve the AWS WAF of one URL")
    solve.add_argument("url")
    solve.add_argument("--proxy", help="http://user:pass@host:port or scheme:host:port:user:pass")
    solve.add_argument("--user-agent")
    args = parser.parse_args(argv)
    _logging(args.log_level)

    if args.cmd == "serve":
        import uvicorn
        from .config import Settings
        from .server import create_app
        settings = Settings.from_env()
        settings.open_access = settings.open_access or args.open
        uvicorn.run(create_app(settings), host=args.host, port=args.port,
                    log_level=args.log_level.lower())
        return 0

    from .waf.errors import SolverError
    from .waf.http import parse_proxy
    from .waf.solver import SolveRequest, SolverSettings, solve_website
    started = time.monotonic()
    try:
        outcome = solve_website(
            SolveRequest(website_url=args.url, proxy_url=parse_proxy(args.proxy),
                         user_agent=args.user_agent),
            SolverSettings(), deadline=started + 180)
    except SolverError as exc:
        print(json.dumps({"errorCode": exc.code, "errorDescription": exc.description}))
        return 1
    print(json.dumps({"cookie": outcome.token, "userAgent": outcome.user_agent,
                      "captcha": outcome.used_captcha, "rounds": outcome.captcha_rounds,
                      "seconds": round(time.monotonic() - started, 2)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
