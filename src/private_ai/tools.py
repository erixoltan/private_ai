"""The abilities a generic agent has: long-term memory, recall of earlier
conversations, and access to local files with the user's permission.

File rules:
* Reading needs the user's approval, once per file or for a whole folder for
  the rest of the session. The prompt says where the contents will go.
* Writing needs approval every single time.
* The config folder, which holds API keys, is never readable or writable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from private_ai.providers import Provider, config_path
from private_ai.store import Store

MAX_FILE_CHARS = 100_000
MAX_DIR_ENTRIES = 500


class UI(Protocol):
    """What an interface (terminal, web) must provide to the agent."""

    def text(self, delta: str) -> None: ...  # a piece of the model's reply

    def notice(self, message: str) -> None: ...  # tool activity and warnings

    def ask(self, question: str, choices: dict[str, str]) -> str: ...  # returns a key of choices


def _tool(name: str, description: str, **params: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
                "required": list(params),
            },
        },
    }


TOOLS = [
    _tool(
        "remember",
        "Save a lasting fact or preference about the user so it is available in every future conversation.",
        fact="One self-contained sentence.",
    ),
    _tool(
        "forget",
        "Delete remembered facts that contain the given text.",
        text="Text identifying the fact(s) to delete.",
    ),
    _tool(
        "search_conversations",
        "Keyword search over the user's earlier conversations. Use when the user refers to something discussed before.",
        query="Keywords to look for.",
    ),
    _tool("read_file", "Read a text file on the user's computer. The user is asked for permission.", path="File path."),
    _tool(
        "list_directory",
        "List the files in a folder on the user's computer. The user is asked for permission.",
        path="Folder path.",
    ),
    _tool(
        "write_file",
        "Create or replace a text file on the user's computer. The user is asked for permission every time.",
        path="File path.",
        content="The complete new contents of the file.",
    ),
]


class ToolError(Exception):
    """A tool could not do what was asked; the message goes back to the model."""


class Toolbox:
    def __init__(self, store: Store, ui: UI, provider: Provider):
        self.store = store
        self.ui = ui
        self.provider = provider
        self.conversation_id: str | None = None
        self.readable_dirs: set[Path] = set()  # folders approved for this session

    def run(self, name: str, arguments: str) -> str:
        """Run one tool call and return the text result for the model."""
        try:
            args = json.loads(arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError
        except ValueError:
            return "Error: tool arguments were not a JSON object."
        handler = getattr(self, f"_{name}", None) if name in {t["function"]["name"] for t in TOOLS} else None
        if handler is None:
            return f"Error: unknown tool '{name}'."
        try:
            return handler(**{k: str(v) for k, v in args.items()})
        except ToolError as e:
            return f"Error: {e}"
        except TypeError:
            return f"Error: wrong arguments for {name}."
        except OSError as e:
            return f"Error: {e.strerror or e}"

    # ---- memory ----

    def _remember(self, fact: str) -> str:
        self.store.remember(fact)
        self.ui.notice(f"remembered: {fact}")
        return "Saved."

    def _forget(self, text: str) -> str:
        removed = self.store.forget(text)
        for fact in removed:
            self.ui.notice(f"forgot: {fact}")
        return f"Deleted {len(removed)} fact(s)." if removed else "Nothing matched."

    def _search_conversations(self, query: str) -> str:
        hits = self.store.search(query, exclude_id=self.conversation_id)
        self.ui.notice(f"searched earlier conversations for: {query} ({len(hits)} found)")
        if not hits:
            return "No earlier conversation mentions that."
        return "\n\n".join(
            f"[{h['date']}, conversation \"{h['conversation']}\", {h['role']}]\n{h['excerpt']}" for h in hits
        )

    # ---- files ----

    def _resolve(self, path: str) -> Path:
        resolved = Path(path).expanduser().resolve()  # follows symlinks, so approval is for the real target
        protected = config_path().resolve().parent
        if resolved == protected or protected in resolved.parents:
            raise ToolError("that folder holds API keys and is off limits.")
        return resolved

    def _destination(self) -> str:
        if self.provider.is_local:
            return "The contents stay on this machine (local model)."
        return f"The contents will be sent to {self.provider.name} (cloud)."

    def _approve_read(self, path: Path, folder: Path, what: str) -> None:
        if any(d == path or d in path.parents for d in self.readable_dirs):
            return
        answer = self.ui.ask(
            f"Allow the assistant to {what} {path}?\n{self._destination()}",
            {"y": "yes, once", "a": f"yes, and everything in {folder} for this session", "n": "no"},
        )
        if answer == "a":
            self.readable_dirs.add(folder)
        elif answer != "y":
            raise ToolError("the user did not allow this.")

    def _read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise ToolError(f"{target} is not a file.")
        self._approve_read(target, target.parent, "read")
        data = target.read_bytes()
        if b"\0" in data[:4096]:
            raise ToolError("that is a binary file; only text files can be read.")
        text = data.decode("utf-8", errors="replace")
        self.ui.notice(f"read {target}")
        if len(text) > MAX_FILE_CHARS:
            return text[:MAX_FILE_CHARS] + f"\n[truncated: file is {len(text)} characters]"
        return text

    def _list_directory(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_dir():
            raise ToolError(f"{target} is not a folder.")
        self._approve_read(target, target, "list")
        entries = sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        self.ui.notice(f"listed {target}")
        lines = [e.name + ("/" if e.is_dir() else "") for e in entries[:MAX_DIR_ENTRIES]]
        if len(entries) > MAX_DIR_ENTRIES:
            lines.append(f"[{len(entries) - MAX_DIR_ENTRIES} more not shown]")
        return "\n".join(lines) or "(empty folder)"

    def _write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        if target.is_dir():
            raise ToolError(f"{target} is a folder.")
        action = "REPLACE the existing file" if target.exists() else "create the new file"
        answer = self.ui.ask(
            f"Allow the assistant to {action} {target}? ({len(content)} characters)",
            {"y": "yes", "n": "no"},
        )
        if answer != "y":
            raise ToolError("the user did not allow this.")
        target.write_text(content, encoding="utf-8")
        self.ui.notice(f"wrote {target}")
        return "Written."
