"""Terminal chat interface."""

from __future__ import annotations

import re
import shutil
import sys
import time
import unicodedata
from collections.abc import Callable

from private_ai import research, search
from private_ai.agent import Agent, pick_model, ready
from private_ai.client import ProviderError
from private_ai.providers import Provider
from private_ai.store import Store

try:
    import readline  # noqa: F401  (line editing and history for input())
except ImportError:
    pass

HELP = """\
/new                 start a new conversation
/list                show saved conversations
/resume [N]          continue conversation N from /list (default: the latest)
/delete N            permanently delete conversation N from /list
/move N              move the latest question and its answer to conversation N
/move new            ... or to a new conversation
/memory              show what is remembered about you
/forget TEXT         delete remembered facts containing TEXT
/provider NAME [MODEL]   switch provider (keeps the conversation)
/model NAME          switch model
/search [NAME]       show the web search services, or switch to one
/research            list saved research; /research N prints report N
/research plan N     print the plan (outline and reasoning) behind report N
/research TOPIC      research a topic on the web and save a report
/quit                leave (Ctrl-D also works)"""


_LIST_MARKER = re.compile(r"[-*•]|\d+[.)]")


def _width(text: str) -> int:
    """Columns the text occupies; CJK and similar characters take two."""
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


class WordWrapper:
    """Wraps streamed text at word boundaries.

    Text arrives in arbitrary pieces, so a word is held back until it is
    complete and then placed on whichever line it fits. The terminal width is
    read as each word is placed, so resizing the window affects what follows.
    Wrapped lines keep the indentation of the line they continue, and wrapped
    list items line up under the item's text.
    """

    def __init__(self, write: Callable[[str], None], width: Callable[[], int]):
        self.write = write
        self.width = width
        self.column = 0
        self.indent = ""  # leading whitespace of the current line
        self.spaces = ""  # whitespace seen since the last word
        self.word = ""

    def feed(self, text: str) -> None:
        for char in text:
            if char == "\n":
                self._place_word()
                self.write("\n")
                self.column, self.indent, self.spaces = 0, "", ""
            elif char in " \t":
                self._place_word()
                self.spaces += char
            else:
                self.word += char

    def flush(self) -> None:
        """Emit a word still being held back (call when the reply ends)."""
        self._place_word()

    def _place_word(self) -> None:
        if not self.word:
            return
        if self.column == 0:
            self.indent = self.spaces
            if _LIST_MARKER.fullmatch(self.word):  # continuation lines align under the item's text
                self.indent += " " * (_width(self.word) + 1)
        elif self.column + _width(self.spaces + self.word) > self.width():
            self.write("\n")
            self.column, self.spaces = 0, self.indent
        self.write(self.spaces + self.word)
        self.column += _width(self.spaces + self.word)
        self.spaces, self.word = "", ""


class TerminalUI:
    def __init__(self):
        self.color = sys.stdout.isatty()
        self.wrapper = WordWrapper(sys.stdout.write, lambda: shutil.get_terminal_size().columns - 1)
        self.in_thinking = False

    def _dim(self, s: str) -> str:
        return f"\033[2m{s}\033[0m" if self.color else s

    def end_line(self) -> None:
        self.wrapper.flush()
        if self.wrapper.column:
            self.wrapper.feed("\n")
        if self.in_thinking:
            sys.stdout.write("\033[0m")
            self.in_thinking = False
        sys.stdout.flush()

    def thinking(self, delta: str) -> None:
        if not self.color:
            return  # piped output gets the reply only
        if not self.in_thinking:
            self.end_line()
            sys.stdout.write("\033[2m")
            self.in_thinking = True
        self.wrapper.feed(delta)
        sys.stdout.flush()

    def text(self, delta: str) -> None:
        if self.in_thinking:
            self.end_line()
        if self.color:  # only wrap for a real terminal, never for piped output
            self.wrapper.feed(delta)
        else:
            self.wrapper.write(delta)
            self.wrapper.column = 0 if delta.endswith("\n") else 1
        sys.stdout.flush()

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


def _banner(agent: Agent) -> None:
    p = agent.provider
    where = "local, nothing leaves this machine" if p.is_local else "cloud"
    print(f"{p.name} ({where}) · {agent.model}")
    if not p.is_local and p.policy_note:
        print(f"  {p.policy_note} Not verified by this program: {p.policy_url}")
    _search_line()


def _search_line() -> None:
    try:
        s = search.current()
    except ValueError as e:
        print(f"web search: {e}")
        return
    print(f"web search: {s.name}. You are asked before each query is sent to {s.destination}.")


def _label(conversation: dict) -> str:
    return f"{conversation.get('emoji', '')} {conversation['title']}".strip()


def _show_history(conversation: dict, ui: TerminalUI) -> None:
    print(ui._dim(f"--- continuing: {_label(conversation)} ---"))
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
        provider = ready(providers[provider_name])
        conversation = None
        if resume:
            saved = store.list()
            conversation = saved[0] if resume == "latest" and saved else None
            if resume != "latest":
                conversation = next((c for c in saved if c["id"] == resume), None)
            if conversation is None:
                raise ProviderError(f"no saved conversation to resume ({resume})")
        agent = Agent(provider, pick_model(provider, model), store, ui, conversation)
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
                        print(f"{i:3}  {when}  {_label(c)}")
                    if not listed:
                        print("No saved conversations yet.")
                elif command == "/resume":
                    listed = listed or store.list()[:20]
                    index = int(arg or 1) - 1
                    if not 0 <= index < len(listed):
                        raise ValueError("no such conversation; see /list")
                    agent = Agent(agent.provider, agent.model, store, ui, listed[index])
                    _show_history(agent.conversation, ui)
                elif command == "/delete":
                    index = int(arg or 0) - 1
                    if not listed or not 0 <= index < len(listed):
                        raise ValueError("run /list first, then /delete N")
                    doomed = listed.pop(index)
                    store.delete(doomed["id"])
                    print(f"Deleted: {_label(doomed)}")
                    if doomed["id"] == agent.conversation["id"]:
                        agent = Agent(agent.provider, agent.model, store, ui)
                        print("New conversation.")
                elif command == "/move":
                    index = int(arg or 0) - 1 if arg != "new" else None
                    if index is not None and (not listed or not 0 <= index < len(listed)):
                        raise ValueError("run /list first, then /move N (or /move new)")
                    target_id = listed[index]["id"] if index is not None else None
                    try:
                        target, emptied = store.move_exchange(agent.conversation["id"], -1, target_id)
                    except OSError:
                        raise ValueError("nothing in this conversation to move yet") from None
                    print(f"Moved the latest exchange to: {_label(target) or '(new conversation)'}")
                    listed = []
                    if emptied:
                        agent = Agent(agent.provider, agent.model, store, ui)
                        print("That left this conversation empty, so it was removed. New conversation.")
                    else:
                        agent = Agent(agent.provider, agent.model, store, ui, store.load(agent.conversation["id"]))
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
                    new = ready(providers[name])
                    agent = Agent(new, pick_model(new, new_model.strip() or None), store, ui, agent.conversation)
                    _banner(agent)
                elif command == "/research":
                    jobs = store.list_research()
                    if not arg:
                        print("\n".join(f"{n:>2}. {research.summary_line(j)}" for n, j in enumerate(jobs, 1)) or "No research yet.")
                    elif arg.removeprefix("plan ").strip().isdigit():
                        number = int(arg.removeprefix("plan "))
                        if not 0 < number <= len(jobs):
                            raise ValueError("no such research; see /research")
                        job = jobs[number - 1]
                        if arg.startswith("plan"):
                            print(job.get("outline") or "(no plan was written)")
                        else:
                            print(job["report"] or f"(no report: {job.get('error') or research.status(job)})")
                    else:
                        backend = search.current()
                        print(research.describe(backend))
                        try:
                            job = research.Research(agent.provider, agent.model, store, backend, ui.notice).run(arg)
                        except KeyboardInterrupt:
                            print("(stopped; what was gathered is saved)")
                            continue
                        print("\n" + job["report"] if job["status"] == "done" else f"error: {job.get('error') or job['status']}")
                        print(ui._dim("(saved; the assistant can use it in any conversation)"))
                elif command == "/search":
                    if arg:
                        search.select(arg)
                        _search_line()
                    else:
                        using = search.current().name
                        for s in search.load_backends().values():
                            state = "in use" if s.name == using else ("ready" if s.ready else f"needs {s.key_env}")
                            print(f"  {s.name:<10} {state:<22} {s.policy_note}")
                elif command == "/model":
                    if not arg:
                        raise ValueError("usage: /model NAME")
                    agent.model = pick_model(agent.provider, arg)
                    _banner(agent)
                else:
                    print("Unknown command. Type /help.")
            except (ValueError, ProviderError) as e:
                print(f"error: {e}")
            continue

        labelled = bool(agent.conversation.get("emoji"))
        try:
            agent.send(line)
            if not labelled and agent.conversation.get("emoji"):
                ui.notice(f"this conversation is now: {_label(agent.conversation)}")
        except ProviderError as e:
            ui.end_line()
            print(f"error: {e}")
        except KeyboardInterrupt:
            ui.end_line()
            print("(interrupted; that message was not saved)")
        ui.end_line()
        print()
