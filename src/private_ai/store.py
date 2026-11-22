"""Everything the agent remembers, kept in plain files on this machine.

    ~/.local/share/private_ai/           (or $PRIVATE_AI_DATA)
        memory.md                        one remembered fact per line; edit freely
        conversations/<id>.json          one file per conversation

Files are created readable only by the user.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from pathlib import Path


def data_dir() -> Path:
    override = os.environ.get("PRIVATE_AI_DATA")
    if override:
        return Path(override)
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "private_ai"


def _write_private(path: Path, text: str) -> None:
    """Atomic write, readable only by the user."""
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


class Store:
    def __init__(self, root: Path | None = None):
        self.root = root or data_dir()
        self.conversations = self.root / "conversations"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.conversations.mkdir(exist_ok=True, mode=0o700)
        self.memory_path = self.root / "memory.md"

    # ---- conversations ----

    def new_conversation(self) -> dict:
        now = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        return {
            "id": f"{stamp}-{secrets.token_hex(2)}",
            "title": "",
            "created": now,
            "updated": now,
            "messages": [],  # OpenAI chat format, so any provider can continue it
        }

    def save(self, conversation: dict) -> None:
        conversation["updated"] = time.time()
        if not conversation["title"]:
            first = next((m["content"] for m in conversation["messages"] if m["role"] == "user"), "")
            conversation["title"] = " ".join(first.split())[:60]
        path = self.conversations / f"{conversation['id']}.json"
        _write_private(path, json.dumps(conversation, indent=1, ensure_ascii=False))

    def load(self, conversation_id: str) -> dict:
        path = self.conversations / f"{conversation_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        """All conversations, most recently updated first."""
        found = []
        for path in self.conversations.glob("*.json"):
            try:
                found.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue  # a damaged file should not hide the others
        return sorted(found, key=lambda c: c["updated"], reverse=True)

    def search(self, query: str, exclude_id: str | None = None, limit: int = 8) -> list[dict]:
        """Keyword search over what was said in earlier conversations."""
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        if not words:
            return []
        hits = []
        for conv in self.list():
            if conv["id"] == exclude_id:
                continue
            for m in conv["messages"]:
                content = m.get("content") or ""
                if m["role"] not in ("user", "assistant") or not content:
                    continue
                lower = content.lower()
                score = sum(w in lower for w in words)
                if score:
                    at = max(0, min(lower.find(w) for w in words if w in lower) - 150)
                    hits.append(
                        {
                            "score": score,
                            "conversation": conv["title"],
                            "date": time.strftime("%Y-%m-%d", time.localtime(conv["updated"])),
                            "role": m["role"],
                            "excerpt": content[at : at + 600],
                        }
                    )
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:limit]

    # ---- long-term memory ----

    def memory(self) -> list[str]:
        if not self.memory_path.exists():
            return []
        lines = self.memory_path.read_text(encoding="utf-8").splitlines()
        return [line.removeprefix("- ").strip() for line in lines if line.strip()]

    def remember(self, fact: str) -> None:
        fact = " ".join(fact.split())
        facts = self.memory()
        if fact and fact not in facts:
            self._write_memory(facts + [fact])

    def forget(self, text: str) -> list[str]:
        """Remove every remembered fact containing the text; returns what was removed."""
        facts = self.memory()
        removed = [f for f in facts if text.lower() in f.lower()] if text.strip() else []
        if removed:
            self._write_memory([f for f in facts if f not in removed])
        return removed

    def _write_memory(self, facts: list[str]) -> None:
        _write_private(self.memory_path, "".join(f"- {f}\n" for f in facts))
