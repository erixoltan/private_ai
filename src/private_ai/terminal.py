"""Terminal chat interface."""

from __future__ import annotations

import sys
import time

from private_ai.agent import Agent
from private_ai.client import ProviderError, check_model, local_models
from private_ai.providers import Provider, config_path
from private_ai.store import Store

try:
    import readline  # noqa: F401  (line editing and history for input())
except ImportError:
    pass

HELP = """\
/new                 start a new conversation
/list                show saved conversations
/resume [N]          continue conversation N from /list (default: the latest)
/memory              show what is remembered about you
/forget TEXT         delete remembered facts containing TEXT
/provider NAME [MODEL]   switch provider (keeps the conversation)
/model NAME          switch model
/quit                leave (Ctrl-D also works)"""


class TerminalUI:
    def __init__(self):
        self.color = sys.stdout.isatty()
        self.mid_line = False

    def _dim(self, s: str) -> str:
        return f"\033[2m{s}\033[0m" if self.color else s

    def end_line(self) -> None:
        if self.mid_line:
            print()
            self.mid_line = False

    def text(self, delta: str) -> None:
        sys.stdout.write(delta)
        sys.stdout.flush()
        self.mid_line = not delta.endswith("\n")

    def notice(self, message: str) -> None:
        self.end_line()
        print(self._dim(f"  [{message}]"))

    def ask(self, question: str, choices: dict[str, str]) -> str:
        self.end_line()
        print(f"\n{question}")
        for key, label in choices.items():
            print(f"  {key}) {label}")
        while True:
            try:
                answer = input(f"[{'/'.join(choices)}] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return "n"
            if answer in choices:
                return answer


def _pick_model(provider: Provider, model: str | None) -> str:
    model = model or provider.model
    if model:
        return check_model(provider, model)  # fail now, not after the user's first message
    models = local_models(provider)
    if not models:
        raise ProviderError(f"{provider.name} lists no models; pass --model")
    return models[0]


def _ready(provider: Provider) -> Provider:
    if provider.needs_key and not provider.api_key:
        raise ProviderError(f"no API key for {provider.name}: set {provider.key_env} or add it to {config_path()}")
    return provider


def _banner(agent: Agent) -> None:
    p = agent.provider
    where = "local, nothing leaves this machine" if p.is_local else "cloud"
    print(f"{p.name} ({where}) · {agent.model}")
    if not p.is_local and p.policy_note:
        print(f"  {p.policy_note} Not verified by this program: {p.policy_url}")


def _show_history(conversation: dict, ui: TerminalUI) -> None:
    print(ui._dim(f"--- continuing: {conversation['title']} ---"))
    shown = [m for m in conversation["messages"] if m["role"] in ("user", "assistant") and m.get("content")]
    for m in shown[-6:]:
        prefix = "you> " if m["role"] == "user" else ""
        print(ui._dim(prefix + m["content"]))
    print()


def run_chat(providers: dict[str, Provider], provider_name: str, model: str | None, resume: str | None) -> int:
    store = Store()
    ui = TerminalUI()
    try:
        if provider_name not in providers:
            raise ProviderError(f"unknown provider '{provider_name}'; known: {', '.join(providers)}")
        provider = _ready(providers[provider_name])
        conversation = None
        if resume:
            saved = store.list()
            conversation = saved[0] if resume == "latest" and saved else None
            if resume != "latest":
                conversation = next((c for c in saved if c["id"] == resume), None)
            if conversation is None:
                raise ProviderError(f"no saved conversation to resume ({resume})")
        agent = Agent(provider, _pick_model(provider, model), store, ui, conversation)
    except ProviderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    _banner(agent)
    print("Type /help for commands.\n")
    if conversation:
        _show_history(conversation, ui)

    listed: list[dict] = []
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\n(Ctrl-D or /quit to leave)")
            continue
        if not line:
            continue

        if line.startswith("/"):
            command, _, arg = line.partition(" ")
            arg = arg.strip()
            try:
                if command in ("/quit", "/exit"):
                    return 0
                elif command == "/help":
                    print(HELP)
                elif command == "/new":
                    agent = Agent(agent.provider, agent.model, store, ui)
                    print("New conversation.")
                elif command == "/list":
                    listed = store.list()[:20]
                    for i, c in enumerate(listed, 1):
                        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(c["updated"]))
                        print(f"{i:3}  {when}  {c['title']}")
                    if not listed:
                        print("No saved conversations yet.")
                elif command == "/resume":
                    listed = listed or store.list()[:20]
                    index = int(arg or 1) - 1
                    if not 0 <= index < len(listed):
                        raise ValueError("no such conversation; see /list")
                    agent = Agent(agent.provider, agent.model, store, ui, listed[index])
                    _show_history(agent.conversation, ui)
                elif command == "/memory":
                    facts = store.memory()
                    print("\n".join(f"- {f}" for f in facts) or "Nothing remembered yet.")
                    print(ui._dim(f"(stored in {store.memory_path})"))
                elif command == "/forget":
                    removed = store.forget(arg)
                    print("\n".join(f"forgot: {f}" for f in removed) or "Nothing matched.")
                elif command == "/provider":
                    name, _, new_model = arg.partition(" ")
                    if name not in providers:
                        raise ValueError(f"known providers: {', '.join(providers)}")
                    new = _ready(providers[name])
                    agent = Agent(new, _pick_model(new, new_model.strip() or None), store, ui, agent.conversation)
                    _banner(agent)
                elif command == "/model":
                    if not arg:
                        raise ValueError("usage: /model NAME")
                    agent.model = _pick_model(agent.provider, arg)
                    _banner(agent)
                else:
                    print("Unknown command. Type /help.")
            except (ValueError, ProviderError) as e:
                print(f"error: {e}")
            continue

        try:
            agent.send(line)
        except ProviderError as e:
            ui.end_line()
            print(f"error: {e}")
        except KeyboardInterrupt:
            ui.end_line()
            print("(interrupted; that message was not saved)")
        ui.end_line()
        print()
