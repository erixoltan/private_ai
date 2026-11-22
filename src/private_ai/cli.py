"""Command line: chat with the agent in the terminal or a local web page, list providers, run connectivity checks."""

from __future__ import annotations

import argparse
import sys
import time

from private_ai.client import ProviderError, chat, local_models
from private_ai.providers import Provider, config_path, config_warnings, default_provider_name, load_providers

PING = [{"role": "user", "content": "Reply with the single word: pong"}]


def check(provider: Provider, model: str | None, timeout: float) -> bool:
    """Two-step check: list models (reachability + auth), then one tiny chat."""
    where = "local" if provider.is_local else "cloud"
    print(f"{provider.name} ({where}) {provider.base_url}")
    if provider.needs_key and not provider.api_key:
        print(f"  SKIP  no API key: set {provider.key_env} or api_key in {config_path()}")
        return False

    model = model or provider.model
    try:
        models = local_models(provider, timeout)
        print(f"  ok    reachable, {len(models)} models available")
        if not model and models:
            model = models[0]
    except ProviderError as e:
        # Some endpoints don't offer a model list; the chat below still decides.
        print(f"  warn  model list failed: {e}")
    if not model:
        print("  FAIL  no model to test with; pass --model or set one in the config")
        return False

    try:
        start = time.monotonic()
        # Generous limit: reasoning models spend tokens before the visible reply.
        msg = chat(provider, PING, model=model, max_tokens=256, timeout=timeout)
        elapsed = time.monotonic() - start
    except ProviderError as e:
        print(f"  FAIL  chat with {model}: {e}")
        return False
    reply = (msg.get("content") or "").strip() or "(empty reply)"
    print(f"  ok    chat with {model} in {elapsed:.1f}s: {reply[:60]!r}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="private-ai", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_chat = sub.add_parser("chat", help="talk to the agent in this terminal")
    p_chat.add_argument("--provider", help="provider to use (default: default_provider in config, else local ollama)")
    p_chat.add_argument("--model", help="model to use (overrides config)")
    p_chat.add_argument(
        "--resume", nargs="?", const="latest", metavar="ID", help="continue the latest conversation, or the one with this id"
    )
    p_web = sub.add_parser("web", help="serve a web interface on this machine only, until you press Ctrl-C")
    p_web.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1 (default: 8765)")
    p_web.add_argument(
        "--detached-timeout",
        type=float,
        default=120,
        metavar="SECONDS",
        help="how long a reply may keep going after its browser tab is closed (default: 120)",
    )
    sub.add_parser("providers", help="show configured providers and their data policies")
    p_check = sub.add_parser("check", help="test connectivity to providers")
    p_check.add_argument("names", nargs="*", help="providers to check (default: all)")
    p_check.add_argument("--model", help="model to test with (overrides config)")
    p_check.add_argument("--timeout", type=float, default=60, help="seconds per request")
    args = parser.parse_args(argv)

    try:
        providers = load_providers()
    except (ValueError, OSError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    for warning in config_warnings():
        print(f"warning: {warning}", file=sys.stderr)

    if args.command == "chat":
        from private_ai.terminal import run_chat

        return run_chat(providers, args.provider or default_provider_name(), args.model, args.resume)

    if args.command == "web":
        from private_ai.web import run_web

        return run_web(providers, default_provider_name(), args.port, args.detached_timeout)

    if args.command == "providers":
        for p in providers.values():
            key = "not needed" if not p.needs_key else ("set" if p.api_key else f"missing ({p.key_env})")
            print(f"{p.name}\n  url:    {p.base_url}\n  key:    {key}\n  model:  {p.model or '(first listed)'}")
            if p.policy_note:
                print(f"  policy: {p.policy_note}")
            if p.policy_url:
                print(f"          {p.policy_url}")
        return 0

    unknown = [n for n in args.names if n not in providers]
    if unknown:
        print(f"unknown provider(s): {', '.join(unknown)}; known: {', '.join(providers)}", file=sys.stderr)
        return 2
    selected = args.names or list(providers)
    results = {name: check(providers[name], args.model, args.timeout) for name in selected}
    passed = sum(results.values())
    print(f"\n{passed}/{len(results)} providers passed")
    return 0 if passed == len(results) else 1
