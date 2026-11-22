"""The web interface's server, driven over real HTTP against a scripted fake model."""

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from private_ai.providers import Provider
from private_ai.store import Store
from private_ai.web import COOKIE, App, Turn, TurnStopped, make_server
from tests.test_agent import FakeModel, HTTPServer


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


class WebTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_server = HTTPServer(("127.0.0.1", 0), FakeModel)
        threading.Thread(target=cls.model_server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.model_server.shutdown()
        cls.model_server.server_close()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name).resolve()
        self.store = Store(self.dir / "data")
        providers = {
            "fake": Provider(name="fake", base_url=f"http://127.0.0.1:{self.model_server.server_port}/v1", model="m"),
            "nokey": Provider(name="nokey", base_url="https://example.invalid/v1", key_env="NO_SUCH_KEY_VAR"),
        }
        self.app = App(providers, "fake", self.store, detached_timeout=30)
        self.server = make_server(self.app, 0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        FakeModel.script, FakeModel.requests = [], []
        FakeModel.side_replies, FakeModel.side_requests = [], []

    def request(self, path, method="GET", body=None, headers=None, auth=True):
        headers = dict(headers or {})
        if auth:
            headers["Cookie"] = f"{COOKIE}={self.server.token}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.build_opener(NoRedirect).open(req, timeout=10) as resp:
                return resp.status, resp.read(), resp.headers
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read(), e.headers

    def get_json(self, path, **kw):
        status, body, _ = self.request(path, **kw)
        return status, json.loads(body)

    def events(self, conversation_id):
        _, body, _ = self.request(f"/api/conversations/{conversation_id}/events")
        return [json.loads(line) for line in body.decode().splitlines() if line.strip()]

    def chat(self, text, conversation_id=None):
        status, data = self.get_json(
            "/api/chat", method="POST", body={"conversation_id": conversation_id, "provider": "fake", "model": "m", "text": text}
        )
        self.assertEqual(status, 200, data)
        return data["conversation_id"]

    # ---- who may use it ----

    def test_refused_without_the_secret(self):
        for path in ("/", "/app.js", "/api/state"):
            self.assertEqual(self.request(path, auth=False)[0], 403)
        self.assertEqual(self.request("/api/state", headers={"Cookie": f"{COOKIE}=wrong"}, auth=False)[0], 403)

    def test_link_from_terminal_sets_cookie_and_hides_the_secret(self):
        status, _, headers = self.request(f"/?token={self.server.token}", auth=False)
        self.assertEqual((status, headers["Location"]), (303, "/"))
        self.assertIn(f"{COOKIE}={self.server.token}", headers["Set-Cookie"])
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        self.assertIn("SameSite=Strict", headers["Set-Cookie"])
        self.assertEqual(self.request("/?token=wrong", auth=False)[0], 403)

    def test_refused_for_foreign_host_or_cross_site_request(self):
        self.assertEqual(self.request("/api/state", headers={"Host": "evil.example:8765"})[0], 403)
        status, _, _ = self.request("/api/chat", method="POST", body={}, headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        own = {"Origin": self.base}
        self.assertEqual(self.request("/api/memory/forget", method="POST", body={}, headers=own)[0], 200)

    def test_page_is_served_with_a_policy_blocking_outside_resources(self):
        status, body, headers = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b"<title>private_ai</title>", body)
        self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
        for name in ("app.js", "app.css"):
            self.assertEqual(self.request("/" + name)[0], 200)
        for name in ("index.html", "app.js", "app.css"):
            text = (Path(__file__).parent.parent / "src/private_ai/web_static" / name).read_text()
            self.assertNotRegex(text, r"(src|href)=\"https?://")
            self.assertNotIn("innerHTML", text)

    # ---- chatting ----

    def test_chat_streams_thinking_and_reply_then_saves(self):
        FakeModel.script = [{"thinking": "hmm", "text": "hello there"}]
        cid = self.chat("hi")
        events = self.events(cid)
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds[0], "thinking")
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "text"), "hello there")
        _, conv = self.get_json(f"/api/conversations/{cid}")
        self.assertEqual([b["type"] for b in conv["blocks"]], ["user", "thinking", "text"])
        self.assertEqual((conv["provider"], conv["model"], conv["running"]), ("fake", "m", False))
        _, state = self.get_json("/api/state")
        self.assertEqual([(c["title"], c["running"]) for c in state["conversations"]], [("hi", False)])

    def test_look_and_summary_marker_reach_the_page(self):
        FakeModel.script = ["hello"]
        FakeModel.side_replies = ['{"title": "Robot chat", "emoji": "\U0001f916", "color": "gray"}']
        cid = self.chat("hi")
        self.events(cid)
        _, conv = self.get_json(f"/api/conversations/{cid}")
        self.assertEqual((conv["title"], conv["emoji"], conv["color"]), ("Robot chat", "\U0001f916", "gray"))
        row = self.get_json("/api/state")[1]["conversations"][0]
        self.assertEqual((row["title"], row["emoji"], row["color"]), ("Robot chat", "\U0001f916", "gray"))
        saved = self.store.load(cid)
        saved["summary"], saved["summary_upto"] = "notes", 2
        saved["messages"] += [{"role": "user", "content": "later"}]
        self.store.save(saved)
        blocks = self.get_json(f"/api/conversations/{cid}")[1]["blocks"]
        self.assertEqual([b["type"] for b in blocks], ["user", "text", "notice", "user"])

    def test_thinking_is_saved_but_never_sent_back(self):
        FakeModel.script = [{"thinking": "private reasoning", "text": "one"}, "two"]
        cid = self.chat("first")
        self.events(cid)
        self.events(self.chat("second", cid))
        self.assertEqual(self.store.load(cid)["messages"][1]["thinking"], "private reasoning")
        self.assertNotIn("private reasoning", json.dumps(FakeModel.requests[-1]))
        self.assertEqual(len(FakeModel.requests[-1]["messages"]), 4)

    def test_permission_question_is_answered_from_the_page(self):
        note = self.dir / "note.txt"
        note.write_text("plum")
        FakeModel.script = [("read_file", {"path": str(note)}), "It says plum."]
        cid = self.chat("read it")
        seen = []

        def watch():
            seen.extend(self.events(cid))

        watcher = threading.Thread(target=watch)
        watcher.start()
        deadline = time.time() + 5
        question = None
        while time.time() < deadline and question is None:
            question = next((e for e in list(self.app.turns[cid].events) if e["type"] == "ask"), None)
            time.sleep(0.02)
        self.assertIn("stay on this machine", question["question"])
        _, conv = self.get_json(f"/api/conversations/{cid}")
        self.assertEqual((conv["running"], conv["pending"], conv["blocks"]), (True, "read it", []))
        listed = self.get_json("/api/state")[1]["conversations"]  # findable before its first save
        self.assertEqual([(c["id"], c["title"], c["running"]) for c in listed], [(cid, "read it", True)])
        self.assertEqual(self.chat_status("again", cid), 409)
        status, _ = self.get_json(f"/api/conversations/{cid}/answer", method="POST", body={"id": question["id"], "answer": "y"})
        self.assertEqual(status, 200)
        watcher.join(5)
        self.assertIn({"type": "answered", "id": question["id"], "answer": "y"}, seen)
        self.assertEqual(FakeModel.requests[-1]["messages"][-1]["content"], "plum")

    def chat_status(self, text, cid):
        return self.get_json("/api/chat", method="POST", body={"conversation_id": cid, "provider": "fake", "text": text})[0]

    def test_question_is_denied_when_nobody_is_watching(self):
        import private_ai.web as web

        self.addCleanup(setattr, web, "ASK_GRACE", web.ASK_GRACE)
        web.ASK_GRACE = 0.2
        note = self.dir / "note.txt"
        note.write_text("plum")
        FakeModel.script = [("read_file", {"path": str(note)}), "I could not read it."]
        cid = self.chat("read it")
        turn = self.app.turns[cid]
        deadline = time.time() + 5
        while not turn.done and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(turn.done)
        self.assertEqual(FakeModel.requests[-1]["messages"][-1]["content"], "Error: the user did not allow this.")

    def test_provider_error_is_reported_and_nothing_saved(self):
        FakeModel.script = [500]
        cid = self.chat("hi")
        self.assertEqual(self.events(cid)[-1]["type"], "error")
        self.assertEqual(self.get_json(f"/api/conversations/{cid}")[0], 404)

    def test_bad_requests(self):
        post = lambda body: self.get_json("/api/chat", method="POST", body=body)[0]
        self.assertEqual(post({"provider": "fake", "text": "  "}), 400)
        self.assertEqual(post({"provider": "nope", "text": "x"}), 400)
        self.assertEqual(post({"provider": "nokey", "text": "x"}), 400)
        self.assertEqual(post({"provider": "fake", "text": "x", "conversation_id": "missing"}), 404)
        self.assertEqual(self.get_json("/api/conversations/..%2F..%2Fetc")[0], 404)

    # ---- housekeeping ----

    def test_move_exchange(self):
        FakeModel.script = ["a1", "a2", "b1"]
        a = self.chat("qa1")
        self.events(a)
        self.events(self.chat("qa2", a))
        b = self.chat("qb1")
        self.events(b)
        move = lambda cid, body: self.get_json(f"/api/conversations/{cid}/move", method="POST", body=body)
        self.assertEqual(move(a, {"exchange": 1, "target": b}), (200, {"target_id": b, "source_deleted": False}))
        texts = lambda cid: [x["text"] for x in self.get_json(f"/api/conversations/{cid}")[1]["blocks"]]
        self.assertEqual((texts(a), texts(b)), (["qa1", "a1"], ["qb1", "b1", "qa2", "a2"]))
        status, data = move(a, {"exchange": 0, "target": None})
        self.assertEqual((status, data["source_deleted"]), (200, True))
        self.assertEqual(texts(data["target_id"]), ["qa1", "a1"])
        self.assertEqual(self.get_json(f"/api/conversations/{a}")[0], 404)
        self.assertEqual(move(b, {"exchange": 9, "target": None})[0], 400)
        self.assertEqual(move(b, {"exchange": -1, "target": None})[0], 400)
        self.assertEqual(move(b, {"exchange": 0, "target": "missing"})[0], 404)
        self.assertEqual(move(b, {"exchange": 0, "target": "../x"})[0], 404)

    def test_delete_conversation(self):
        FakeModel.script = ["hello"]
        cid = self.chat("hi")
        self.events(cid)
        self.assertEqual(self.request(f"/api/conversations/{cid}", method="DELETE")[0], 200)
        self.assertEqual(self.get_json(f"/api/conversations/{cid}")[0], 404)
        self.assertEqual(self.store.list(), [])

    def test_memory_view_and_forget(self):
        self.store.remember("Likes tea.")
        self.store.remember("Likes tea with milk.")
        self.assertEqual(self.get_json("/api/memory")[1]["facts"], ["Likes tea.", "Likes tea with milk."])
        _, data = self.get_json("/api/memory/forget", method="POST", body={"fact": "Likes tea."})
        self.assertEqual(data["facts"], ["Likes tea with milk."])

    def test_state_lists_providers_and_models(self):
        _, state = self.get_json("/api/state")
        by_name = {p["name"]: p for p in state["providers"]}
        self.assertEqual((by_name["fake"]["local"], by_name["fake"]["ready"]), (True, True))
        self.assertEqual((by_name["nokey"]["local"], by_name["nokey"]["ready"]), (False, False))
        self.assertEqual(self.get_json("/api/models?provider=nokey")[0], 400)
        self.assertEqual(self.get_json("/api/models?provider=fake")[1], {"models": ["glm", "x/routers/Kimi-1", "x/models/kimi-2", "m"]})


class TurnTest(unittest.TestCase):
    def test_stops_when_cancelled_or_left_unwatched_too_long(self):
        turn = Turn("x", 0, detached_timeout=0.05)
        turn.attach()
        time.sleep(0.1)
        turn.check()  # watched: carries on however long it takes
        turn.detach()
        turn.check()  # just left: still within the allowance
        time.sleep(0.1)
        with self.assertRaises(TurnStopped):
            turn.check()
        watched = Turn("x", 0, detached_timeout=60)
        watched.attach()
        watched.cancelled = True
        with self.assertRaises(TurnStopped):
            watched.check()


if __name__ == "__main__":
    unittest.main()
