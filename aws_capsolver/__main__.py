"""Command line: `serve` runs the API, `solve` solves one URL and prints the token,
`billing` manages token balances (top-ups after a payment, history, prices)."""

import argparse
import json
import logging
import os
import sys
import time


def _logging(level: str) -> None:
    from .waf.http import install_log_redaction
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    install_log_redaction()  # proxy passwords never reach the log output


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
    bill = sub.add_parser("billing", help="token balances: topup / balance / history / prices / accounts")
    bill.add_argument("action", choices=["topup", "balance", "history", "prices", "accounts"])
    bill.add_argument("--key", help="client key")
    amount = bill.add_mutually_exclusive_group()
    amount.add_argument("--usd", type=float, help="package bought, converted at tokens_per_usd")
    amount.add_argument("--tokens", type=int, help="exact tokens (negative = correction)")
    bill.add_argument("--note", default="", help="free text stored in the ledger (e.g. payment id)")
    bill.add_argument("--limit", type=int, default=30)
    args = parser.parse_args(argv)
    _logging(args.log_level)

    if args.cmd == "billing":
        return _billing(args)

    if args.cmd == "serve":
        import uvicorn
        from .config import Settings
        from .server import create_app
        settings = Settings.from_env()
        settings.open_access = settings.open_access or args.open
        # one process only: the task registry lives in memory (scale with AWSCS_WORKERS)
        uvicorn.run(create_app(settings), host=args.host, port=args.port,
                    log_level=args.log_level.lower(), log_config=None)
        return 0

    from .config import Settings
    from .waf import pow as powlib
    from .waf.errors import SolverError
    from .waf.http import parse_proxy
    from .waf.solver import SolveRequest, SolverSettings, solve_website
    settings = Settings.from_env()
    started = time.monotonic()
    try:
        outcome = solve_website(
            SolveRequest(website_url=args.url, proxy_url=parse_proxy(args.proxy),
                         user_agent=args.user_agent),
            SolverSettings(http_timeout=settings.http_timeout,
                           pow_processes=settings.pow_processes,
                           captcha_rounds=settings.captcha_rounds,
                           verify_token=settings.verify_token,
                           allow_private=settings.allow_private),
            deadline=started + max(settings.task_timeout, 30))
    except SolverError as exc:
        print(json.dumps({"errorCode": exc.code, "errorDescription": exc.description}))
        return 1
    finally:
        powlib.shutdown_pool()
    print(json.dumps({"cookie": outcome.token, "userAgent": outcome.user_agent,
                      "captcha": outcome.used_captcha, "rounds": outcome.captcha_rounds,
                      "seconds": round(time.monotonic() - started, 2)}))
    return 0


def _billing(args) -> int:
    from .config import KeyStore, Settings
    from .server import make_billing
    settings = Settings.from_env()
    settings.billing = True
    keys = KeyStore(settings)
    billing = make_billing(settings, keys)
    try:
        if args.action == "prices":
            print(json.dumps({"tokens_per_usd": billing.tokens_per_usd, "prices": billing.prices}, indent=1))
            return 0
        if args.action == "accounts":
            print(json.dumps(billing.accounts(), indent=1))
            return 0
        if not args.key:
            print("--key is required", file=sys.stderr)
            return 2
        if keys.lookup(args.key) is None:
            print(f"unknown client key (add it to {settings.keys_file} first)", file=sys.stderr)
            return 2
        if args.action == "topup":
            if args.usd is None and args.tokens is None:
                print("give --usd or --tokens", file=sys.stderr)
                return 2
            try:
                new = billing.topup(args.key, tokens=args.tokens, usd=args.usd, note=args.note)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            print(json.dumps({"key": args.key, "tokens": new}))
        elif args.action == "balance":
            tokens = billing.balance(args.key)
            print(json.dumps({"key": args.key, "tokens": tokens,
                              "usd": round(tokens / billing.tokens_per_usd, 4)}))
        else:
            print(json.dumps(billing.history(args.key, args.limit), indent=1))
        return 0
    finally:
        billing.close()


if __name__ == "__main__":
    sys.exit(main())
