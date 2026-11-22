"""Agent, tools and storage, tested against a scripted fake model on localhost."""

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from private_ai.agent import Agent
from private_ai.client import ProviderError
from private_ai.providers import Provider
from private_ai.store import Store
from private_ai.tools import Toolbox


class FakeModel(BaseHTTPRequestHandler):
    """Replies from a script: each entry is text, or a (tool, args) pair, or an HTTP error code."""

    script: list = []
    requests: list = []

    def do_GET(self):  # /api/tags: not an Ollama server
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        FakeModel.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        step = FakeModel.script.pop(0)
        if isinstance(step, int):
            body = b'{"error": "this model does not support tools"}'
            self.send_response(step)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if isinstance(step, str):
            deltas = [{"content": step[:2]}, {"content": step[2:]}]
        else:  # tool call with its arguments split across chunks, as real providers do
            name, args = step[0], json.dumps(step[1])
            deltas = [
                {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": name, "arguments": args[:3]}}]},
                {"tool_calls": [{"index": 0, "function": {"arguments": args[3:]}}]},
            ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for d in deltas:
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': d}]})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


class FakeUI:
    def __init__(self, answers=()):
        self.answers = list(answers)
        self.shown = ""
        self.notices = []
        self.questions = []

    def text(self, delta):
        self.shown += delta

    def notice(self, message):
        self.notices.append(message)

    def ask(self, question, choices):
        self.questions.append(question)
        return self.answers.pop(0)


class AgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeModel)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.provider = Provider(name="fake", base_url=f"http://127.0.0.1:{cls.server.server_port}/v1")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name).resolve()
        self.store = Store(self.dir / "data")
        FakeModel.script, FakeModel.requests = [], []

    def agent(self, ui, conversation=None):
        return Agent(self.provider, "m", self.store, ui, conversation)

    def test_plain_reply_is_streamed_and_saved(self):
        FakeModel.script = ["hello there"]
        ui = FakeUI()
        agent = self.agent(ui)
        agent.send("hi")
        self.assertEqual(ui.shown, "hello there")
        saved = self.store.load(agent.conversation["id"])
        self.assertEqual([m["role"] for m in saved["messages"]], ["user", "assistant"])
        self.assertEqual(saved["title"], "hi")
        mode = os.stat(self.store.conversations / f"{saved['id']}.json").st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_remember_tool_then_memory_reaches_next_conversation(self):
        FakeModel.script = [("remember", {"fact": "Prefers metric units."}), "Noted.", "ok"]
        self.agent(FakeUI()).send("I prefer metric")
        self.assertEqual(self.store.memory(), ["Prefers metric units."])
        self.agent(FakeUI()).send("new conversation")
        self.assertIn("Prefers metric units.", FakeModel.requests[-1]["messages"][0]["content"])

    def test_resume_sends_earlier_messages(self):
        FakeModel.script = ["first reply", "second reply"]
        first = self.agent(FakeUI())
        first.send("one")
        resumed = self.agent(FakeUI(), self.store.load(first.conversation["id"]))
        resumed.send("two")
        sent = [m["content"] for m in FakeModel.requests[-1]["messages"][1:]]
        self.assertEqual(sent, ["one", "first reply", "two"])

    def test_search_finds_other_conversations(self):
        FakeModel.script = ["The capital of Mongolia is Ulaanbaatar."]
        self.agent(FakeUI()).send("capital of Mongolia?")
        hits = self.store.search("mongolia capital")
        self.assertEqual(len(hits), 2)
        self.assertIn("Ulaanbaatar", hits[0]["excerpt"] + hits[1]["excerpt"])

    def test_failed_turn_is_rolled_back(self):
        FakeModel.script = [500]
        agent = self.agent(FakeUI())
        with self.assertRaises(ProviderError):
            agent.send("hi")
        self.assertEqual(agent.conversation["messages"], [])

    def test_model_without_tool_support_falls_back_to_plain_chat(self):
        FakeModel.script = [400, "plain answer"]
        ui = FakeUI()
        self.agent(ui).send("hi")
        self.assertEqual(ui.shown, "plain answer")
        self.assertNotIn("tools", FakeModel.requests[-1])

    def test_file_read_through_agent(self):
        note = self.dir / "note.txt"
        note.write_text("the secret word is plum")
        FakeModel.script = [("read_file", {"path": str(note)}), "It says plum."]
        ui = FakeUI(answers=["y"])
        self.agent(ui).send("read my note")
        self.assertIn("stay on this machine", ui.questions[0])
        tool_result = FakeModel.requests[-1]["messages"][-1]
        self.assertEqual((tool_result["role"], tool_result["content"]), ("tool", "the secret word is plum"))


class FileToolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name).resolve()
        (self.dir / "a.txt").write_text("A")
        (self.dir / "b.txt").write_text("B")
        self.cloud = Provider(name="cloudy", base_url="https://example.invalid/v1")

    def box(self, answers, provider=None):
        self.ui = FakeUI(answers)
        return Toolbox(Store(self.dir / "data"), self.ui, provider or self.cloud)

    def call(self, box, name, **args):
        return box.run(name, json.dumps(args))

    def test_denied_read_returns_nothing(self):
        result = self.call(self.box(["n"]), "read_file", path=str(self.dir / "a.txt"))
        self.assertEqual(result, "Error: the user did not allow this.")
        self.assertIn("sent to cloudy (cloud)", self.ui.questions[0])

    def test_once_asks_again_but_folder_grant_does_not(self):
        box = self.box(["y", "a"])
        self.assertEqual(self.call(box, "read_file", path=str(self.dir / "a.txt")), "A")
        self.assertEqual(self.call(box, "read_file", path=str(self.dir / "b.txt")), "B")
        self.assertEqual(self.call(box, "read_file", path=str(self.dir / "a.txt")), "A")
        self.assertEqual(len(self.ui.questions), 2)

    def test_folder_grant_does_not_follow_symlink_out(self):
        outside = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        outside.write("outside")
        outside.close()
        self.addCleanup(os.unlink, outside.name)
        (self.dir / "link.txt").symlink_to(outside.name)
        box = self.box(["a", "n"])
        self.call(box, "list_directory", path=str(self.dir))
        result = self.call(box, "read_file", path=str(self.dir / "link.txt"))
        self.assertEqual(result, "Error: the user did not allow this.")

    def test_write_always_asks(self):
        box = self.box(["a", "y", "n"])
        self.call(box, "list_directory", path=str(self.dir))
        new = self.dir / "new.txt"
        self.assertEqual(self.call(box, "write_file", path=str(new), content="x"), "Written.")
        self.assertIn("did not allow", self.call(box, "write_file", path=str(new), content="y"))
        self.assertEqual(new.read_text(), "x")
        self.assertIn("REPLACE", self.ui.questions[-1])

    def test_config_folder_is_off_limits(self):
        os.environ["PRIVATE_AI_CONFIG"] = str(self.dir / "cfg" / "config.toml")
        self.addCleanup(os.environ.pop, "PRIVATE_AI_CONFIG")
        (self.dir / "cfg").mkdir()
        (self.dir / "cfg" / "config.toml").write_text('KEY = "secret"')
        box = self.box([])
        for name, args in [("read_file", {"path": str(self.dir / "cfg" / "config.toml")}),
                           ("list_directory", {"path": str(self.dir / "cfg")}),
                           ("write_file", {"path": str(self.dir / "cfg" / "x"), "content": ""})]:
            self.assertIn("off limits", self.call(box, name, **args))
        self.assertEqual(self.ui.questions, [])

    def test_bad_tool_calls_are_reported_not_raised(self):
        box = self.box([])
        self.assertIn("unknown tool", box.run("run", "{}"))
        self.assertIn("not a JSON object", box.run("read_file", "{bad"))
        self.assertIn("wrong arguments", box.run("read_file", "{}"))


if __name__ == "__main__":
    unittest.main()
