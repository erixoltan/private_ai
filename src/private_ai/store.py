"""Everything the agent remembers, kept in plain files on this machine.

    ~/.local/share/private_ai/           (or $PRIVATE_AI_DATA)
        memory.md                        one remembered fact per line; edit freely
        conversations/<id>.json          one file per conversation
        pages/<hash>.json                web pages the agent has read, with address and date
        research/<id>.json               research reports, with their notes and sources

Files are created readable only by the user.
"""

from __future__ import annotations

import fcntl
import hashlib
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
        self.pages = self.root / "pages"
        self.pages.mkdir(exist_ok=True, mode=0o700)
        self.research = self.root / "research"
        self.research.mkdir(exist_ok=True, mode=0o700)
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

    def _path(self, conversation_id: str) -> Path:
        # Ids can arrive from a URL, so never let one name a file elsewhere.
        if not re.fullmatch(r"[\w-]+", conversation_id):
            raise FileNotFoundError(conversation_id)
        return self.conversations / f"{conversation_id}.json"

    def save(self, conversation: dict) -> None:
        conversation["updated"] = time.time()
        if not conversation["title"]:
            first = next((m["content"] for m in conversation["messages"] if m["role"] == "user"), "")
            conversation["title"] = " ".join(first.split())[:60]
        _write_private(self._path(conversation["id"]), json.dumps(conversation, indent=1, ensure_ascii=False))

    def load(self, conversation_id: str) -> dict:
        return json.loads(self._path(conversation_id).read_text(encoding="utf-8"))

    def delete(self, conversation_id: str) -> None:
        self._path(conversation_id).unlink(missing_ok=True)

    def move_exchange(self, source_id: str, number: int, target_id: str | None = None) -> tuple[dict, bool]:
        """Move one exchange (a user message and everything up to the next one)
        to the end of another conversation, or to a new one if target_id is None.

        number counts exchanges from 0; -1 is the latest. Returns the target
        conversation and whether the source, left empty, was deleted.
        """
        source = self.load(source_id)
        messages = source["messages"]
        starts = [i for i, m in enumerate(messages) if m["role"] == "user"]
        try:
            start = starts[number]
        except IndexError:
            raise ValueError("no such exchange") from None
        end = next((i for i in starts if i > start), len(messages))
        if target_id == source_id:
            raise ValueError("that is the conversation it is already in")
        target = self.load(target_id) if target_id else self.new_conversation()

        target["messages"] += messages[start:end]
        self.save(target)  # target first: an interruption leaves a duplicate, never a loss
        del messages[start:end]
        if start < source.get("summary_upto", 0):
            # The summary describes what just left; drop it and let it be rebuilt.
            source.pop("summary", None)
            source.pop("summary_upto", None)
        if messages:
            self.save(source)
        else:
            self.delete(source_id)
        return target, not messages

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

    def forget(self, text: str, exact: bool = False) -> list[str]:
        """Remove every remembered fact containing the text (or equal to it,
        with exact=True); returns what was removed."""
        facts = self.memory()
        if exact:
            removed = [f for f in facts if f == text]
        else:
            removed = [f for f in facts if text.lower() in f.lower()] if text.strip() else []
        if removed:
            self._write_memory([f for f in facts if f not in removed])
        return removed

    def _write_memory(self, facts: list[str]) -> None:
        _write_private(self.memory_path, "".join(f"- {f}\n" for f in facts))

    # ---- web pages the agent has read ----

    def _page_path(self, url: str) -> Path:
        return self.pages / (hashlib.sha256(url.encode()).hexdigest()[:20] + ".json")

    def save_page(self, url: str, title: str, text: str) -> None:
        """Keep the latest copy of a page, so it can be found and reread later."""
        page = {"url": url, "title": title, "fetched": time.time(), "text": text}
        _write_private(self._page_path(url), json.dumps(page, indent=1, ensure_ascii=False))

    def load_page(self, url: str) -> dict | None:
        try:
            return json.loads(self._page_path(url).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def search_pages(self, query: str, limit: int = 6) -> list[dict]:
        """Keyword search over saved pages: url, title, date and an excerpt around the first match."""
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        hits = []
        for path in self.pages.glob("*.json") if words else []:
            try:
                page = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            lower = f"{page['title']}\n{page['text']}".lower()
            score = sum(w in lower for w in words)
            if score:
                at = max(0, min(page["text"].lower().find(w) for w in words if w in lower) - 200)
                hits.append(
                    {
                        "score": score,
                        "fetched": page["fetched"],
                        "url": page["url"],
                        "title": page["title"],
                        "date": time.strftime("%Y-%m-%d", time.localtime(page["fetched"])),
                        "excerpt": page["text"][at : at + 800],
                    }
                )
        hits.sort(key=lambda h: (h["score"], h["fetched"]), reverse=True)
        return hits[:limit]

    # ---- research reports ----

    def new_research(self, topic: str) -> dict:
        now = time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        return {
            "id": f"{stamp}-{secrets.token_hex(2)}",
            "topic": topic,
            "status": "running",  # then done, failed or stopped; see research.status for "interrupted"
            "created": now,
            "finished": None,
            "queries": [],
            "sources": [],  # n, url, title, date, notes
            "unread": [],  # pages that refused: url, title, problem
            "log": [],
            "outline": "",  # the plan for the report: sections, points and the reasoning behind them
            "report": "",
        }

    def _research_path(self, research_id: str) -> Path:
        if not re.fullmatch(r"[\w-]+", research_id):
            raise ValueError("bad research id")
        return self.research / f"{research_id}.json"

    def save_research(self, job: dict) -> None:
        job["updated"] = time.time()
        _write_private(self._research_path(job["id"]), json.dumps(job, indent=1, ensure_ascii=False))

    def load_research(self, research_id: str) -> dict:
        return json.loads(self._research_path(research_id).read_text(encoding="utf-8"))

    def delete_research(self, research_id: str) -> None:
        self._research_path(research_id).unlink(missing_ok=True)
        self._research_path(research_id).with_suffix(".lock").unlink(missing_ok=True)

    def claim_research(self, research_id: str):
        """Mark a job as being worked on by this process. Returns something to
        close when the work ends, or None if the job is already being worked on.

        It is a lock on a file, which the system releases by itself if the
        process dies, so a job that was cut off never looks as if it is running.
        """
        handle = open(self._research_path(research_id).with_suffix(".lock"), "a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return None
        return handle

    def research_claimed(self, research_id: str) -> bool:
        """Whether some running program, this one or another, is working on the job."""
        if not self._research_path(research_id).with_suffix(".lock").exists():
            return False
        handle = self.claim_research(research_id)
        if handle is None:
            return True
        handle.close()
        return False

    def list_research(self) -> list[dict]:
        """All research jobs, newest first."""
        found = []
        for path in self.research.glob("*.json"):
            try:
                found.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(found, key=lambda j: j["created"], reverse=True)

    def search_research(self, query: str, limit: int = 2) -> list[dict]:
        """Finished reports that mention the query's words, best match first."""
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        scored = []
        for job in self.list_research():
            if job["status"] != "done":
                continue
            topic, report = job["topic"].lower(), job["report"].lower()
            score = sum(3 * (w in topic) + (w in report) for w in words)
            if score:
                scored.append((score, job["created"], job))
        scored.sort(key=lambda s: s[:2], reverse=True)
        return [job for _, _, job in scored[:limit]]
