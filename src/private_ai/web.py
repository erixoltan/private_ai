"""Local web interface.

The server exists only while `private-ai web` is running in a terminal, and it
is built so that nobody but the person who started it can use it:

* It listens on 127.0.0.1 only, never on the network.
* Every request needs a secret that is generated at startup and handed over
  through the link printed in the terminal.
* Requests naming an unexpected host, or sent by another site's page, are
  refused, so a web page open in the same browser cannot drive the agent.
* The page loads nothing from the internet, and a content security policy
  makes the browser enforce that.

A reply keeps being generated if the browser tab goes away, and is saved when
it finishes, but only for a limited time (detached_timeout).
"""

from __future__ import annotations

import hmac
import json
import re
import secrets
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from private_ai.agent import Agent, pick_model, ready
from private_ai.client import ProviderError, local_models
from private_ai.providers import Provider
from private_ai.store import Store

STATIC = Path(__file__).with_name("web_static")
STATIC_TYPES = {"index.html": "text/html", "app.js": "text/javascript", "app.css": "text/css"}
COOKIE = "private_ai_token"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
ASK_GRACE = 5.0  # seconds a permission question survives without a page to answer it
MAX_BODY = 1_000_000


class TurnStopped(Exception):
    """The user pressed stop, or left and the reply took too long."""


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Turn:
    """One exchange in progress: the events produced so far, and who is watching.

    Events are kept so a reloaded page can replay the turn from the start.
    """

    def __init__(self, user_text: str, start_len: int, detached_timeout: float):
        self.user_text = user_text
        self.start_len = start_len  # messages in the conversation before this turn
        self.detached_timeout = detached_timeout
        self.events: list[dict] = []
        self.changed = threading.Condition()
        self.done = False
        self.cancelled = False
        self.listeners = 0
        self.detached_at = time.monotonic()
        self.questions: dict[str, list] = {}  # id -> [threading.Event, answer]

    def emit(self, **event) -> None:
        with self.changed:
            self.events.append(event)
            self.changed.notify_all()

    def finish(self) -> None:
        with self.changed:
            self.done = True
            self.changed.notify_all()

    def attach(self) -> None:
        with self.changed:
            self.listeners += 1

    def detach(self) -> None:
        with self.changed:
            self.listeners -= 1
            self.detached_at = time.monotonic()

    def unwatched_for(self) -> float:
        with self.changed:
            return 0.0 if self.listeners else time.monotonic() - self.detached_at

    def check(self) -> None:
        if self.cancelled or self.unwatched_for() > self.detached_timeout:
            raise TurnStopped


class WebUI:
    """Turns the agent's output into events for the page (see tools.UI)."""

    def __init__(self, turn: Turn):
        self.turn = turn

    def text(self, delta: str) -> None:
        self.turn.check()
        self.turn.emit(type="text", text=delta)

    def thinking(self, delta: str) -> None:
        self.turn.check()
        self.turn.emit(type="thinking", text=delta)

    def notice(self, message: str) -> None:
        self.turn.emit(type="notice", text=message)

    def ask(self, question: str, choices: dict[str, str]) -> str:
        question_id = secrets.token_hex(4)
        answered = threading.Event()
        self.turn.questions[question_id] = [answered, "n"]
        self.turn.emit(type="ask", id=question_id, question=question, choices=choices)
        while not answered.wait(0.25):
            # Nobody there to answer means no.
            if self.turn.cancelled or self.turn.unwatched_for() > ASK_GRACE:
                break
        answer = self.turn.questions.pop(question_id)[1]
        self.turn.emit(type="answered", id=question_id, answer=answer)
        return answer


def _display(messages: list[dict], summary_upto: int = 0) -> list[dict]:
    """Saved messages as the blocks the page shows."""
    blocks = []
    for index, m in enumerate(messages):
        if index == summary_upto and index:
            blocks.append({"type": "notice", "text": "the model now sees a summary of everything above this line"})
        if m["role"] == "user":
            blocks.append({"type": "user", "text": m["content"]})
        elif m["role"] == "assistant":
            if m.get("thinking"):
                blocks.append({"type": "thinking", "text": m["thinking"]})
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for call in m.get("tool_calls") or []:
                try:
                    args = json.loads(call["function"]["arguments"])
                    detail = next((str(args[k]) for k in ("path", "query", "fact", "text") if k in args), "")
                except (ValueError, TypeError, AttributeError):
                    detail = ""
                blocks.append({"type": "notice", "text": f"{call['function']['name']}: {detail[:200]}"})
    return blocks


class App:
    """Everything the page can ask for. Independent of HTTP."""

    def __init__(
        self, providers: dict[str, Provider], default_provider: str, store: Store, detached_timeout: float = 120
    ):
        self.providers = providers
        self.default_provider = default_provider
        self.store = store
        self.detached_timeout = detached_timeout
        self.lock = threading.Lock()
        self.agents: dict[str, Agent] = {}  # kept so folder approvals last while the server runs
        self.turns: dict[str, Turn] = {}

    def _running(self, conversation_id: str) -> Turn | None:
        turn = self.turns.get(conversation_id)
        return turn if turn and not turn.done else None

    def state(self) -> dict:
        return {
            "default_provider": self.default_provider,
            "providers": [
                {
                    "name": p.name,
                    "local": p.is_local,
                    "ready": not p.needs_key or bool(p.api_key),
                    "model": p.model,
                    "policy_note": p.policy_note,
                    "policy_url": p.policy_url,
                }
                for p in self.providers.values()
            ],
            "conversations": self._conversation_list(),
        }

    def _conversation_list(self) -> list[dict]:
        saved = self.store.list()
        known = {c["id"] for c in saved}
        # A new conversation is not on disk until its first reply is complete.
        unsaved = [
            {"id": cid, "title": " ".join(turn.user_text.split())[:60], "updated": time.time()}
            for cid, turn in list(self.turns.items())
            if not turn.done and cid not in known
        ]
        rows = unsaved + [
            {"id": c["id"], "title": c["title"], "updated": c["updated"], "emoji": c.get("emoji"), "color": c.get("color")}
            for c in saved
        ]
        return [{**row, "running": bool(self._running(row["id"]))} for row in rows]

    def models(self, provider_name: str) -> dict:
        provider = self._provider(provider_name)
        try:
            # By name, ignoring any path prefix, so a family (glm, kimi, ...) sits together.
            return {"models": sorted(local_models(provider), key=lambda m: (m.rsplit("/", 1)[-1].lower(), m))}
        except ProviderError as e:
            raise ApiError(502, str(e)) from None

    def _provider(self, name: str) -> Provider:
        if name not in self.providers:
            raise ApiError(400, f"unknown provider '{name}'")
        try:
            return ready(self.providers[name])
        except ProviderError as e:
            raise ApiError(400, str(e)) from None

    def conversation(self, conversation_id: str) -> dict:
        turn = self._running(conversation_id)
        if turn:  # show it as it was before the turn; the page replays the turn's events
            conv = self.agents[conversation_id].conversation
            messages = conv["messages"][: turn.start_len]
        else:
            try:
                conv = self.store.load(conversation_id)
            except (OSError, ValueError):
                raise ApiError(404, "no such conversation") from None
            messages = conv["messages"]
        return {
            "id": conv["id"],
            "title": conv["title"] or (" ".join(turn.user_text.split())[:60] if turn else ""),
            "emoji": conv.get("emoji"),
            "color": conv.get("color"),
            "provider": conv.get("provider"),
            "model": conv.get("model"),
            "blocks": _display(messages, conv.get("summary_upto", 0)),
            "running": bool(turn),
            "pending": turn.user_text if turn else None,
        }

    def start_turn(self, conversation_id: str | None, provider_name: str, model: str | None, text: str) -> str:
        if not text.strip():
            raise ApiError(400, "empty message")
        provider = self._provider(provider_name)
        try:
            model = pick_model(provider, model)
        except ProviderError as e:
            raise ApiError(400, str(e)) from None

        with self.lock:
            if conversation_id and self._running(conversation_id):
                raise ApiError(409, "this conversation is still answering")
            conversation = None
            if conversation_id:
                try:
                    conversation = self.store.load(conversation_id)  # fresh, in case the terminal added to it
                except (OSError, ValueError):
                    raise ApiError(404, "no such conversation") from None
            agent = self.agents.get(conversation_id) if conversation_id else None
            if agent is None or agent.provider.name != provider.name:
                # A new provider starts with no folder approvals: what was approved
                # for a local model was not approved for sending to a cloud.
                agent = Agent(provider, model, self.store, None, conversation)
            else:
                agent.conversation, agent.model = conversation, model
            conversation_id = agent.conversation["id"]
            turn = Turn(text, len(agent.conversation["messages"]), self.detached_timeout)
            agent.ui = agent.toolbox.ui = WebUI(turn)
            self.agents[conversation_id] = agent
            self.turns[conversation_id] = turn

        threading.Thread(target=self._run, args=(agent, turn, text), daemon=True).start()
        return conversation_id

    def _run(self, agent: Agent, turn: Turn, text: str) -> None:
        try:
            agent.send(text)
        except TurnStopped:
            turn.emit(type="error", text="Stopped. This exchange was not saved.")
        except ProviderError as e:
            turn.emit(type="error", text=str(e))
        except Exception as e:  # keep the server alive and tell the user
            turn.emit(type="error", text=f"internal error: {e!r}")
        finally:
            turn.finish()

    def answer(self, conversation_id: str, question_id: str, answer: str) -> None:
        turn = self._running(conversation_id)
        question = turn.questions.get(question_id) if turn else None
        if question is None:
            raise ApiError(404, "that question is no longer open")
        question[1] = answer
        question[0].set()

    def stop(self, conversation_id: str) -> None:
        turn = self._running(conversation_id)
        if turn:
            turn.cancelled = True

    def move(self, conversation_id: str, number: int, target_id: str | None) -> dict:
        with self.lock:
            if self._running(conversation_id) or (target_id and self._running(target_id)):
                raise ApiError(409, "wait until both conversations have finished answering")
            try:
                target, emptied = self.store.move_exchange(conversation_id, number, target_id)
            except OSError:
                raise ApiError(404, "no such conversation") from None
            except ValueError as e:
                raise ApiError(400, str(e)) from None
            if emptied:
                self.agents.pop(conversation_id, None)
                self.turns.pop(conversation_id, None)
        return {"target_id": target["id"], "source_deleted": emptied}

    def delete(self, conversation_id: str) -> None:
        with self.lock:
            if self._running(conversation_id):
                raise ApiError(409, "this conversation is still answering")
            self.agents.pop(conversation_id, None)
            self.turns.pop(conversation_id, None)
            try:
                self.store.delete(conversation_id)
            except OSError:
                raise ApiError(404, "no such conversation") from None


class Handler(BaseHTTPRequestHandler):
    server_version = "private_ai"
    sys_version = ""

    @property
    def app(self) -> App:
        return self.server.app

    def log_message(self, *args):
        pass  # conversation ids are in URLs; keep them out of the terminal

    # ---- gatekeeping ----

    def _allowed(self) -> bool:
        host = self.headers.get("Host", "")
        if host.rsplit(":", 1)[0] not in LOCAL_HOSTS:
            return False  # a foreign name pointed at this machine (DNS rebinding)
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{host}":
            return False  # sent by another site's page
        cookie = SimpleCookie(self.headers.get("Cookie", "")).get(COOKIE)
        return cookie is not None and hmac.compare_digest(cookie.value, self.server.token)

    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; form-action 'none'",
        )
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ApiError(413, "request too large")
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            raise ApiError(400, "body is not JSON") from None
        if not isinstance(data, dict):
            raise ApiError(400, "body is not a JSON object")
        return data

    def _handle(self, method: str) -> None:
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        try:
            # The link from the terminal: trade the secret for a cookie, then drop it from the URL.
            if method == "GET" and url.path == "/" and "token" in query:
                if not hmac.compare_digest(query["token"][0], self.server.token):
                    raise ApiError(403, "That link is out of date. Use the one printed in the terminal.")
                cookie = f"{COOKIE}={self.server.token}; HttpOnly; SameSite=Strict; Path=/"
                return self._send(303, b"", "text/plain", {"Location": "/", "Set-Cookie": cookie})
            if not self._allowed():
                raise ApiError(403, "Open the link printed in the terminal where private-ai web is running.")
            self._route(method, url.path, query)
        except ApiError as e:
            self._json({"error": str(e)}, e.status)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _route(self, method: str, path: str, query: dict) -> None:
        app = self.app
        static = "index.html" if path == "/" else path.lstrip("/")
        conv = re.fullmatch(r"/api/conversations/([\w-]+)(/events|/answer|/stop|/move)?", path)

        if method == "GET" and static in STATIC_TYPES:
            self._send(200, (STATIC / static).read_bytes(), STATIC_TYPES[static])
        elif method == "GET" and path == "/api/state":
            self._json(app.state())
        elif method == "GET" and path == "/api/models":
            self._json(app.models(query.get("provider", [""])[0]))
        elif method == "GET" and path == "/api/memory":
            self._json({"facts": app.store.memory()})
        elif method == "POST" and path == "/api/memory/forget":
            app.store.forget(str(self._body().get("fact", "")), exact=True)
            self._json({"facts": app.store.memory()})
        elif method == "POST" and path == "/api/chat":
            body = self._body()
            conversation_id = app.start_turn(
                body.get("conversation_id"), str(body.get("provider", "")), body.get("model"), str(body.get("text", ""))
            )
            self._json({"conversation_id": conversation_id})
        elif conv and method == "GET" and conv[2] is None:
            self._json(app.conversation(conv[1]))
        elif conv and method == "DELETE" and conv[2] is None:
            app.delete(conv[1])
            self._json({})
        elif conv and method == "GET" and conv[2] == "/events":
            self._events(app.turns.get(conv[1]))
        elif conv and method == "POST" and conv[2] == "/answer":
            body = self._body()
            app.answer(conv[1], str(body.get("id", "")), str(body.get("answer", "")))
            self._json({})
        elif conv and method == "POST" and conv[2] == "/move":
            body = self._body()
            number, target = body.get("exchange"), body.get("target")
            if type(number) is not int or number < 0 or not (target is None or isinstance(target, str)):
                raise ApiError(400, "bad request")
            self._json(app.move(conv[1], number, target))
        elif conv and method == "POST" and conv[2] == "/stop":
            app.stop(conv[1])
            self._json({})
        else:
            raise ApiError(404, "not found")

    def _events(self, turn: Turn | None) -> None:
        """Stream a turn's events, one JSON object per line, until it is done."""
        if turn is None:
            raise ApiError(404, "nothing is running")
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        sent = 0
        turn.attach()
        try:
            while True:
                with turn.changed:
                    if sent == len(turn.events) and not turn.done:
                        turn.changed.wait(2)
                    batch, finished = turn.events[sent:], turn.done
                sent += len(batch)
                # The blank line when idle is how a closed tab gets noticed.
                data = "".join(json.dumps(e) + "\n" for e in batch) or "\n"
                self.wfile.write(data.encode())
                self.wfile.flush()
                if finished and not batch:
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            turn.detach()

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_DELETE(self):
        self._handle("DELETE")


def make_server(app: App, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.app = app
    server.token = secrets.token_urlsafe(24)
    return server


def run_web(providers: dict[str, Provider], default_provider: str, port: int, detached_timeout: float) -> int:
    app = App(providers, default_provider, Store(), detached_timeout)
    try:
        server = make_server(app, port)
    except OSError as e:
        print(f"error: cannot listen on port {port} ({e.strerror}); try --port with another number")
        return 2
    print("private_ai web interface, for this machine only. Open:\n")
    print(f"  http://127.0.0.1:{server.server_port}/?token={server.token}\n")
    print("The link works only while this program is running. Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. The web interface is no longer reachable.")
    finally:
        server.server_close()
    return 0
