"""Command line: chat with the agent in the terminal or a local web page, list providers, run connectivity checks."""

from __future__ import annotations

import argparse
import sys
import time

from private_ai import approvals, research, search
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


def run_research(providers: dict[str, Provider], args) -> int:
    from private_ai.agent import pick_model, ready
    from private_ai.store import Store

    store = Store()
    jobs = store.list_research()
    chosen = args.show or args.plan or args.retry
    job = None
    if chosen:
        job = jobs[int(chosen) - 1] if chosen.isdigit() and 0 < int(chosen) <= len(jobs) else None
        job = job or next((j for j in jobs if j["id"] == chosen), None)
        if not job:
            print("error: no such research; see --list", file=sys.stderr)
            return 1
    if args.show:
        print(job["report"] or f"(no report: {job.get('error') or research.status(job, store)})")
        return 0
    if args.plan:
        print(job.get("outline") or "(no plan was written)")
        return 0
    if args.retry and (job["report"] or not job.get("gathered")):
        print("error: only research that finished reading but has no report can be retried", file=sys.stderr)
        return 1
    if args.retry and store.research_claimed(job["id"]):
        print("error: that research is still being worked on", file=sys.stderr)
        return 1
    if args.list or not (args.topic or job):
        print("\n".join(f"{n:>2}. {research.summary_line(j, store)}" for n, j in enumerate(jobs, 1)) or "No research yet.")
        return 0
    try:
        provider = ready(providers[args.provider or default_provider_name()])
        model = pick_model(provider, args.model)
        backend = search.current()
    except (KeyError, ValueError, ProviderError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{provider.name} ({'local' if provider.is_local else 'cloud'}) · {model}")
    if not job:
        print(research.describe(backend, args.searches, args.pages) + "\n")
    runner = research.Research(
        provider, model, store, backend, lambda m: print(f"  [{m}]", file=sys.stderr), searches=args.searches, pages=args.pages
    )
    try:
        job = runner.run(job["topic"] if job else args.topic, job)
    except KeyboardInterrupt:
        print("\n(stopped; what was gathered is saved)", file=sys.stderr)
        return 130
    if job["status"] != "done":
        print(f"error: {job.get('error') or job['status']}", file=sys.stderr)
        return 1
    print("\n" + job["report"])
    print(f"\nSaved in {store.research / (job['id'] + '.json')}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="private-ai", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_chat = sub.add_parser("chat", help="talk to the agent in this terminal")
    p_chat.add_argument("--provider", help="provider to use (default: default_provider in config, else local ollama)")
    p_chat.add_argument("--model", help="model to use (overrides config)")
    p_chat.add_argument(
        "--resume", nargs="?", const="latest", metavar="ID", help="continue the latest conversation, or the one with this id"
    )
    p_code = sub.add_parser("code", help="work on the code in the current project, in this terminal")
    p_code.add_argument("--provider", help="provider to use (default: default_provider in config, else local ollama)")
    p_code.add_argument("--model", help="model to use (overrides config)")
    p_code.add_argument(
        "--resume", nargs="?", const="latest", metavar="ID", help="continue the latest conversation, or the one with this id"
    )
    p_code.add_argument(
        "--read",
        action="append",
        default=[],
        metavar="FOLDER",
        help="a folder outside the project that may be read without asking, such as a library the project uses; "
        "give it again for each folder",
    )
    p_web = sub.add_parser("web", help="serve a web interface on this machine only, until you press Ctrl-C")
    p_web.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1 (default: 8765)")
    p_web.add_argument(
        "--detached-timeout",
        type=float,
        default=0,
        metavar="SECONDS",
        help="stop a reply left running this long with no browser tab watching it (default: 0, no limit)",
    )
    for p in (p_chat, p_code, p_web):
        p.add_argument(
            "--trust",
            choices=approvals.TRUST_LEVELS,
            help="what is allowed without asking: ask (nothing; the default), project (reading and changing files "
            "in the project), all (everything, including any command). Default: trust in the config file.",
        )
    sub.add_parser("approvals", help="show the approval rules in force in the current project")
    sub.add_parser("providers", help="show configured providers and search services, and their data policies")
    p_research = sub.add_parser("research", help="research a topic on the web and save a report with sources")
    p_research.add_argument("topic", nargs="?", help="the question to research")
    p_research.add_argument("--list", action="store_true", help="list saved research")
    p_research.add_argument("--show", metavar="N", help="print report N from --list (or its id)")
    p_research.add_argument("--plan", metavar="N", help="print the plan (outline and reasoning) behind report N")
    p_research.add_argument("--retry", metavar="N", help="write the report again for research N whose reading was done")
    p_research.add_argument("--provider", help="provider to use (default: default_provider in config, else local ollama)")
    p_research.add_argument("--model", help="model to use (overrides config)")
    p_research.add_argument("--searches", type=int, default=research.DEFAULT_SEARCHES, help="most searches to send")
    p_research.add_argument("--pages", type=int, default=research.DEFAULT_PAGES, help="most pages to read")
    p_search = sub.add_parser("search", help="run one web search and print the results (a test of a search service)")
    p_search.add_argument("query")
    p_search.add_argument("--with", dest="backend", metavar="NAME", help="search service to use (default: the one chat uses)")
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

    trust = getattr(args, "trust", None) or approvals.default_trust()
    if trust not in approvals.TRUST_LEVELS:
        print(f"config error: trust must be one of: {', '.join(approvals.TRUST_LEVELS)}", file=sys.stderr)
        return 2

    if args.command in ("chat", "code"):
        from private_ai.terminal import run_chat

        coding = args.command == "code"
        return run_chat(
            providers, args.provider or default_provider_name(), args.model, args.resume, coding, trust,
            getattr(args, "read", []),
        )

    if args.command == "web":
        from private_ai.web import run_web

        return run_web(providers, default_provider_name(), args.port, args.detached_timeout, trust)

    if args.command == "approvals":
        policy = approvals.Policy(approvals.find_project(), local=True, trust=trust)
        print(f"project: {policy.project}\ntrust:   {policy.trust}\nrules:   {policy.path}")
        print("\n".join(f"  {r.describe()}" for r in policy.rules() if r.source == "file") or "  (no saved rules)")
        if policy.problem:
            print(f"warning: {policy.problem}", file=sys.stderr)
        return 0

    if args.command == "providers":
        for p in providers.values():
            key = "not needed" if not p.needs_key else ("set" if p.api_key else f"missing ({p.key_env})")
            print(f"{p.name}\n  url:    {p.base_url}\n  key:    {key}\n  model:  {p.model or '(first listed)'}")
            if p.policy_note:
                print(f"  policy: {p.policy_note}")
            if p.policy_url:
                print(f"          {p.policy_url}")
        using = search.default_backend_name()
        print("\nweb search")
        for s in search.load_backends().values():
            key = "not needed" if not s.key_env else ("set" if s.api_key else f"missing ({s.key_env})")
            print(f"{s.name}{'  (default)' if s.name == using else ''}\n  url:    {s.base_url}\n  key:    {key}")
            print(f"  policy: {s.policy_note}\n          {s.policy_url}")
        return 0

    if args.command == "research":
        return run_research(providers, args)

    if args.command == "search":
        try:
            backend = search.select(args.backend) if args.backend else search.current()
            results = search.search(backend, args.query)
        except (ValueError, ProviderError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"{backend.name}: {len(results)} results")
        for r in results:
            print(f"\n{r['title']}  {r['date']}\n  {r['url']}\n  {r['snippet'][:200]}")
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
