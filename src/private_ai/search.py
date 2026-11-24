"""Web search and news, through whichever search service the user chooses.

No single service is built in as "the" search. Each one is a small entry with
its own statement of where a query goes:

    searxng     a SearXNG server you run yourself; no account, no key
    brave       Brave Search API               BRAVE_API_KEY
    ollama      Ollama's web search API        OLLAMA_API_KEY
    wikipedia   Wikipedia only; no account, no key

Configuration, in the same file as the model providers:

    default_search = "searxng"           # otherwise: the first one that is set up
    BRAVE_API_KEY = "..."

    [search.searxng]
    base_url = "http://localhost:8888"

    [search.work]                        # another server of a known kind
    kind = "searxng"
    base_url = "https://search.example.org"

    [news]
    feeds = ["https://feeds.bbci.co.uk/news/world/rss.xml"]

A search query leaves this machine (unless the service runs on it), and it
goes to a company that the model provider's privacy policy does not cover.
The agent therefore asks before sending one; see tools.py.

News is different: the feeds are downloaded whole and filtered here, so the
news sites learn that the feed was read but never what the user asked about.
"""

from __future__ import annotations

import ipaddress
import os
import time
import tomllib
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit

from private_ai.client import ProviderError, _request
from private_ai.fetch import FetchError, fetch, strip_tags
from private_ai.providers import config_path

SNIPPET_CHARS = 600
DEFAULT_FEEDS = (
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.npr.org/1001/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
)


@dataclass(frozen=True)
class SearchBackend:
    name: str
    kind: str  # searxng, brave, ollama or wikipedia
    base_url: str
    key_env: str | None = None
    api_key: str | None = None
    # The service's own published statement. This program cannot verify it.
    policy_note: str | None = None
    policy_url: str | None = None

    @property
    def is_local(self) -> bool:
        host = urlsplit(self.base_url).hostname or ""
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    @property
    def ready(self) -> bool:
        return self.key_env is None or bool(self.api_key)

    @property
    def destination(self) -> str:
        """Where a query goes, for the question put to the user."""
        host = urlsplit(self.base_url).hostname
        if self.kind == "searxng":
            where = "your own SearXNG server on this machine" if self.is_local else f"the SearXNG server at {host}"
            return f"{where}, which passes it to public search engines without your account"
        return f"{host} (cloud)"


_BUILTIN = (
    SearchBackend(
        name="searxng",
        kind="searxng",
        base_url="http://localhost:8888",
        policy_note="Runs on your machine and queries public search engines for you; they see the query but no account.",
        policy_url="https://docs.searxng.org/own-instance.html",
    ),
    SearchBackend(
        name="brave",
        kind="brave",
        base_url="https://api.search.brave.com",
        key_env="BRAVE_API_KEY",
        policy_note="Queries go to Brave under your API key. Read its terms for what is kept.",
        policy_url="https://api-dashboard.search.brave.com/app/documentation",
    ),
    SearchBackend(
        name="ollama",
        kind="ollama",
        base_url="https://ollama.com",
        key_env="OLLAMA_API_KEY",
        policy_note="Queries go to Ollama under your API key. Its documentation does not say what is kept.",
        policy_url="https://docs.ollama.com/capabilities/web-search",
    ),
    SearchBackend(
        name="wikipedia",
        kind="wikipedia",
        base_url="https://en.wikipedia.org",
        policy_note="Searches Wikipedia only. No account; Wikimedia sees the query and your address.",
        policy_url="https://foundation.wikimedia.org/wiki/Policy:Privacy_policy",
    ),
)
_KINDS = {b.kind: b for b in _BUILTIN}


def _config(path: Path | None) -> dict:
    path = path or config_path()
    if path.exists():
        with path.open("rb") as f:
            return tomllib.load(f)
    return {}


def load_backends(path: Path | None = None) -> dict[str, SearchBackend]:
    """Built-in search services merged with the user's config file and environment."""
    config = _config(path)
    backends = {b.name: b for b in _BUILTIN}
    for name, entry in config.get("search", {}).items():
        fields = {k: entry[k] for k in ("base_url", "key_env", "api_key") if k in entry}
        base = backends.get(name) or _KINDS.get(entry.get("kind", ""))
        if base is None:
            raise ValueError(f"search '{name}' needs kind = one of: {', '.join(_KINDS)}")
        backends[name] = replace(base, name=name, **fields)
    for name, b in backends.items():
        key = (os.environ.get(b.key_env) or config.get(b.key_env)) if b.key_env else None
        backends[name] = replace(b, base_url=b.base_url.rstrip("/"), api_key=key or b.api_key)
    return backends


def default_backend_name(path: Path | None = None) -> str:
    """The configured default; else a SearXNG server the user has set up; else
    the first service with a key; else Wikipedia, which needs nothing."""
    config = _config(path)
    backends = load_backends(path)
    chosen = config.get("default_search")
    if chosen:
        if chosen not in backends:
            raise ValueError(f"default_search = '{chosen}' is not a known search; known: {', '.join(backends)}")
        return chosen
    for name in config.get("search", {}):
        if backends[name].ready:
            return name
    return next((name for name, b in backends.items() if b.key_env and b.ready), "wikipedia")


def news_feeds(path: Path | None = None) -> list[str]:
    return list(_config(path).get("news", {}).get("feeds", DEFAULT_FEEDS))


# The choice made with /search or in the web page, for as long as the program runs.
_selected: str | None = None


def select(name: str) -> SearchBackend:
    global _selected
    backends = load_backends()
    if name not in backends:
        raise ValueError(f"known searches: {', '.join(backends)}")
    if not backends[name].ready:
        raise ValueError(f"{name} needs an API key: set {backends[name].key_env} or add it to {config_path()}")
    _selected = name
    return backends[name]


def current() -> SearchBackend:
    backends = load_backends()
    return backends[_selected if _selected in backends else default_backend_name()]


# ---- searching ----


def _clip(text: str) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " ..."


def search(backend: SearchBackend, query: str, count: int = 8, timeout: float = 20) -> list[dict]:
    """Results as dicts with title, url, snippet and, when known, date."""
    if not backend.ready:
        raise ProviderError(f"{backend.name} needs an API key: set {backend.key_env} or add it to {config_path()}")
    results = _SEARCHERS[backend.kind](backend, query, count, timeout)
    return [r for r in results if r.get("url", "").startswith(("http://", "https://"))][:count]


def _searxng(backend: SearchBackend, query: str, count: int, timeout: float) -> list[dict]:
    url = f"{backend.base_url}/search?" + urlencode({"q": query, "format": "json"})
    try:
        data = _request(backend, "", timeout=timeout, url=url)
    except ProviderError as e:
        if "HTTP 403" in str(e):
            raise ProviderError(
                f"the SearXNG server at {backend.base_url} refuses JSON requests. "
                "Add 'json' to search.formats in its settings.yml and restart it."
            ) from None
        if "cannot reach" in str(e):
            raise ProviderError(f"no SearXNG server is answering at {backend.base_url}") from None
        raise
    return [
        {
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "snippet": _clip(r.get("content") or ""),
            "date": (r.get("publishedDate") or "")[:10],
        }
        for r in data.get("results", [])
    ]


def _brave(backend: SearchBackend, query: str, count: int, timeout: float) -> list[dict]:
    url = f"{backend.base_url}/res/v1/web/search?" + urlencode({"q": query, "count": count})
    data = _request(backend, "", timeout=timeout, url=url, key_header="X-Subscription-Token")
    return [
        {
            "title": strip_tags(r.get("title") or ""),
            "url": r.get("url") or "",
            "snippet": _clip(strip_tags(r.get("description") or "")),
            "date": (r.get("page_age") or "")[:10],
        }
        for r in (data.get("web") or {}).get("results", [])
    ]


def _ollama(backend: SearchBackend, query: str, count: int, timeout: float) -> list[dict]:
    body = {"query": query, "max_results": min(count, 10)}
    data = _request(backend, "/api/web_search", body, timeout=timeout)
    return [
        {"title": r.get("title") or "", "url": r.get("url") or "", "snippet": _clip(r.get("content") or ""), "date": ""}
        for r in data.get("results", [])
    ]


def _wikipedia(backend: SearchBackend, query: str, count: int, timeout: float) -> list[dict]:
    params = {"action": "query", "list": "search", "srsearch": query, "srlimit": count, "format": "json", "utf8": 1}
    data = _request(backend, "", timeout=timeout, url=f"{backend.base_url}/w/api.php?" + urlencode(params))
    return [
        {
            "title": r.get("title") or "",
            "url": f"{backend.base_url}/wiki/" + quote((r.get("title") or "").replace(" ", "_")),
            "snippet": _clip(strip_tags(r.get("snippet") or "")),
            "date": (r.get("timestamp") or "")[:10],
        }
        for r in (data.get("query") or {}).get("search", [])
    ]


_SEARCHERS = {"searxng": _searxng, "brave": _brave, "ollama": _ollama, "wikipedia": _wikipedia}


def reader_fetch(backend: SearchBackend, url: str, timeout: float = 30) -> tuple[str, str]:
    """(title, text) of a page, fetched for us by Ollama's servers. They see the address."""
    data = _request(backend, "/api/web_fetch", {"url": url}, timeout=timeout)
    return data.get("title") or "", (data.get("content") or "").strip()


# ---- news feeds ----


def _local(tag: str) -> str:
    return tag.rpartition("}")[2].lower()


def _when(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return 0.0
    try:
        return parsedate_to_datetime(text).timestamp()
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def parse_feed(body: bytes) -> list[dict]:
    """Items of an RSS or Atom feed: source, title, url, snippet, time."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []
    source = ""
    items = []
    for node in root.iter():
        if _local(node.tag) not in ("item", "entry"):
            if _local(node.tag) == "title" and not source:
                source = " ".join((node.text or "").split())
            continue
        item = {"source": source, "title": "", "url": "", "snippet": "", "time": 0.0}
        for child in node:
            name, text = _local(child.tag), (child.text or "").strip()
            if name == "title":
                item["title"] = strip_tags(text)
            elif name == "link":
                # RSS puts the address in the text; Atom in href, with rel="alternate" for the article.
                if text or child.get("rel", "alternate") == "alternate":
                    item["url"] = item["url"] or text or child.get("href", "")
            elif name in ("description", "summary") and not item["snippet"]:
                item["snippet"] = _clip(strip_tags(text))[:300]
            elif name in ("pubdate", "published", "updated", "date"):
                item["time"] = item["time"] or _when(text)
        if item["title"] and item["url"].startswith(("http://", "https://")):
            items.append(item)
    return items


def headlines(feeds: list[str], topic: str = "", limit: int = 30, timeout: float = 15) -> tuple[list[dict], list[str]]:
    """Newest items across the feeds, optionally only those mentioning the topic.
    The topic is matched here, on this machine. Returns (items, problems)."""

    def one(url: str):
        try:
            # The feed list is the user's own, so a feed on the local network is allowed.
            return parse_feed(fetch(url, timeout, allow_private=True, feeds=True)[2])
        except FetchError as e:
            return f"{urlsplit(url).hostname}: {e}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        fetched = list(pool.map(one, feeds))
    problems = [f for f in fetched if isinstance(f, str)]
    items = [item for f in fetched if isinstance(f, list) for item in f]
    words = [w for w in topic.lower().split() if len(w) > 2]
    if words:
        items = [i for i in items if any(w in f"{i['title']} {i['snippet']}".lower() for w in words)]
    seen: set[str] = set()
    unique = [i for i in items if not (i["url"] in seen or seen.add(i["url"]))]
    unique.sort(key=lambda i: i["time"], reverse=True)
    return unique[:limit], problems


def day(timestamp: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(timestamp)) if timestamp else ""
