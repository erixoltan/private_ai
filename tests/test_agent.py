"""Agent, tools and storage, tested against a scripted fake model on localhost."""

import json
import os
import tempfile
import threading
import unittest
import unittest.mock
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
    side_replies: list = []  # answers to non-streaming requests (summaries, titles); "no" when empty
    side_requests: list = []

    def do_GET(self):
        found = self.path == "/v1/models"  # anything else (/api/tags): not an Ollama server
        body = b'{"data": [{"id": "x/models/kimi-2"}, {"id": "m"}, {"id": "x/routers/Kimi-1"}, {"id": "glm"}]}' if found else b""
        self.send_response(200 if found else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not request["stream"]:
            FakeModel.side_requests.append(request)
            reply = FakeModel.side_replies.pop(0) if FakeModel.side_replies else "no"
            body = json.dumps({"choices": [{"message": {"role": "assistant", "content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        FakeModel.requests.append(request)
        step = FakeModel.script.pop(0)
        if isinstance(step, int):
            body = b'{"error": "this model does not support tools"}'
            self.send_response(step)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if isinstance(step, dict):  # a reply preceded by reasoning
            deltas = [{"reasoning_content": step["thinking"]}, {"content": step["text"]}]
        elif isinstance(step, str):
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

    def thinking(self, delta):
        self.thought = getattr(self, "thought", "") + delta

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
        FakeModel.side_replies, FakeModel.side_requests = [], []

    def agent(self, ui, conversation=None, provider=None):
        return Agent(provider or self.provider, "m", self.store, ui, conversation)

    # ---- moving an exchange ----

    def two_conversations(self):
        FakeModel.script = ["a1", "a2", ("remember", {"fact": "Likes plums."}), "a3", "b1"]
        a, b = self.agent(FakeUI()), self.agent(FakeUI())
        for text in ("qa1", "qa2", "qa3"):
            a.send(text)
        b.send("qb1")
        return a.conversation["id"], b.conversation["id"]

    def contents(self, conversation_id):
        return [m.get("content") or m["role"] for m in self.store.load(conversation_id)["messages"]]

    def test_move_exchange_to_another_conversation(self):
        a, b = self.two_conversations()
        target, emptied = self.store.move_exchange(a, 1, b)
        self.assertEqual((target["id"], emptied), (b, False))
        self.assertEqual(self.contents(a), ["qa1", "a1", "qa3", "assistant", "Saved.", "a3"])
        self.assertEqual(self.contents(b), ["qb1", "b1", "qa2", "a2"])

    def test_move_keeps_tool_calls_with_their_exchange_and_latest_is_minus_one(self):
        a, b = self.two_conversations()
        self.store.move_exchange(a, -1, b)
        self.assertEqual(self.contents(a), ["qa1", "a1", "qa2", "a2"])
        self.assertEqual(self.contents(b), ["qb1", "b1", "qa3", "assistant", "Saved.", "a3"])

    def test_move_to_new_conversation_and_emptying_the_source_deletes_it(self):
        a, b = self.two_conversations()
        new, emptied = self.store.move_exchange(a, 0)
        self.assertEqual((self.contents(new["id"]), new["title"], emptied), (["qa1", "a1"], "qa1", False))
        _, emptied = self.store.move_exchange(b, 0, a)
        self.assertTrue(emptied)
        self.assertNotIn(b, [c["id"] for c in self.store.list()])
        self.assertEqual(self.contents(a)[-2:], ["qb1", "b1"])

    def test_move_out_of_summarised_part_discards_the_summary(self):
        a, b = self.two_conversations()
        conv = self.store.load(a)
        conv["summary"], conv["summary_upto"] = "notes mentioning qa1", 4
        self.store.save(conv)
        self.store.move_exchange(a, 2, b)  # after the summarised part: summary stays
        self.assertEqual(self.store.load(a)["summary_upto"], 4)
        self.store.move_exchange(a, 0, b)  # inside it: summary would be wrong now
        self.assertNotIn("summary", self.store.load(a))
        self.assertNotIn("summary_upto", self.store.load(a))

    def test_move_rejects_bad_requests(self):
        a, b = self.two_conversations()
        for args in ((a, 7, b), (a, 0, a)):
            with self.assertRaises(ValueError):
                self.store.move_exchange(*args)
        with self.assertRaises(OSError):
            self.store.move_exchange(a, 0, "missing")
        self.assertEqual(len(self.contents(a)), 8)

    # ---- title, emoji and colour ----

    def test_model_chooses_title_emoji_and_colour_once(self):
        FakeModel.script = ["It depends on the vows.", "second"]
        FakeModel.side_replies = ['Sure! {"title": "Wedding anniversary plans.", "emoji": "\u2764\ufe0f", "color": "Rose"}']
        agent = self.agent(FakeUI())
        agent.send("help me plan our anniversary")
        saved = self.store.load(agent.conversation["id"])
        self.assertEqual((saved["title"], saved["emoji"], saved["color"]), ("Wedding anniversary plans", "\u2764\ufe0f", "rose"))
        self.assertIn("help me plan our anniversary", FakeModel.side_requests[0]["messages"][1]["content"])
        agent.send("more")
        self.assertEqual(len(FakeModel.side_requests), 1)

    def test_unusable_look_is_ignored_and_retried_only_once(self):
        FakeModel.script = ["a", "b", "c"]
        FakeModel.side_replies = ['{"title": "T", "emoji": "robot", "color": "gray"}', '{"title": "T", "emoji": "\U0001f916", "color": "puce"}']
        agent = self.agent(FakeUI())
        for text in ("one", "two", "three"):
            agent.send(text)
        saved = self.store.load(agent.conversation["id"])
        self.assertEqual((saved["title"], saved.get("emoji"), saved.get("color")), ("one", None, None))
        self.assertEqual(len(FakeModel.side_requests), 2)

    # ---- long conversations ----

    def test_long_conversation_is_summarised_for_the_model_but_kept_on_disk(self):
        from dataclasses import replace

        small = replace(self.provider, context_tokens=400)
        FakeModel.script = ["r1 " + "x" * 600, "r2 " + "y" * 600, "r3", "r4"]
        FakeModel.side_replies = ["no", "no", "NOTES: user likes plums", "no"]
        ui = FakeUI()
        agent = self.agent(ui, provider=small)
        agent.send("first question")
        agent.send("second question")
        self.assertEqual(len(FakeModel.requests[-1]["messages"]), 4)  # still sent whole
        agent.send("third question")
        sent = FakeModel.requests[-1]["messages"]
        self.assertEqual([m["content"] for m in sent[1:]], ["third question"])
        self.assertIn("NOTES: user likes plums", sent[0]["content"])
        self.assertIn("summarising", ui.notices[0])
        summary_request = FakeModel.side_requests[2]["messages"][1]["content"]
        self.assertIn("User: first question", summary_request)
        self.assertIn("Assistant: r2", summary_request)
        saved = self.store.load(agent.conversation["id"])
        self.assertEqual(len(saved["messages"]), 6)  # nothing is deleted
        self.assertEqual((saved["summary"], saved["summary_upto"]), ("NOTES: user likes plums", 4))
        # resuming later keeps using the summary
        self.agent(FakeUI(), saved, provider=small).send("fourth question")
        self.assertEqual([m["content"] for m in FakeModel.requests[-1]["messages"][1:]], ["third question", "r3", "fourth question"])

    def test_second_summary_builds_on_the_first(self):
        from dataclasses import replace

        small = replace(self.provider, context_tokens=300)
        FakeModel.script = ["a" * 900, "b" * 900, "c"]
        # in order: look (1st try), summary at q2, look (2nd try), summary at q3
        FakeModel.side_replies = ["no", "FIRST NOTES", "no", "SECOND NOTES"]
        agent = self.agent(FakeUI(), provider=small)
        for text in ("q1", "q2", "q3"):
            agent.send(text)
        self.assertIn("FIRST NOTES", FakeModel.side_requests[3]["messages"][1]["content"])
        self.assertEqual(agent.conversation["summary"], "SECOND NOTES")
        self.assertIn("SECOND NOTES", FakeModel.requests[-1]["messages"][0]["content"])

    def test_tool_exchange_is_never_split_by_a_summary(self):
        from dataclasses import replace

        small = replace(self.provider, context_tokens=200)
        FakeModel.script = ["z" * 900, ("remember", {"fact": "Likes plums."}), "Saved."]
        FakeModel.side_replies = ["no", "NOTES"]
        agent = self.agent(FakeUI(), provider=small)
        agent.send("q1")
        agent.send("remember I like plums")
        roles = [m["role"] for m in FakeModel.requests[-1]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])

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

    def test_write_question_shows_new_contents(self):
        box = self.box(["y"])
        self.call(box, "write_file", path=str(self.dir / "new.txt"), content="first\nsecond\n")
        self.assertIn("Contents:\nfirst\nsecond", self.ui.questions[0])
        self.assertNotIn("WARNING", self.ui.questions[0])

    def test_write_question_shows_changes_to_an_existing_file(self):
        target = self.dir / "c.txt"
        target.write_text("one\ntwo\nthree\n")
        box = self.box(["n", "n"])
        self.call(box, "write_file", path=str(target), content="one\n2\nthree\n")
        self.assertIn("-two\n+2", self.ui.questions[0])
        self.call(box, "write_file", path=str(target), content="one\ntwo\nthree\n")
        self.assertIn("identical", self.ui.questions[1])

    def test_write_question_says_how_much_is_not_shown(self):
        box = self.box(["n"])
        self.call(box, "write_file", path=str(self.dir / "long.txt"), content="x\n" * 200)
        self.assertIn("NOT SHOWN: 120 more lines", self.ui.questions[0])

    def at_home(self):
        home = self.dir / "home"
        (home / ".ssh").mkdir(parents=True)
        (home / ".ssh" / "id_ed25519").write_text("KEY")
        (home / "notes.txt").write_text("N")
        patcher = unittest.mock.patch.dict(os.environ, {"HOME": str(home)})
        patcher.start()
        self.addCleanup(patcher.stop)
        return home

    def test_secrets_are_asked_about_every_time_despite_a_folder_grant(self):
        home = self.at_home()
        box = self.box(["a", "y", "n"])
        self.call(box, "list_directory", path=str(home))
        self.assertEqual(self.call(box, "read_file", path=str(home / "notes.txt")), "N")
        self.assertEqual(self.call(box, "read_file", path=str(home / ".ssh" / "id_ed25519")), "KEY")
        self.assertIn("did not allow", self.call(box, "read_file", path=str(home / ".ssh" / "id_ed25519")))
        self.assertEqual(len(self.ui.questions), 3)
        self.assertIn("WARNING", self.ui.questions[2])
        self.assertIn("sent to cloudy (cloud)", self.ui.questions[2])

    def test_secret_file_names_are_flagged_anywhere(self):
        (self.dir / ".env").write_text("TOKEN=1")
        (self.dir / ".env.example").write_text("TOKEN=")
        box = self.box(["a", "n"])
        self.call(box, "list_directory", path=str(self.dir))
        self.assertEqual(self.call(box, "read_file", path=str(self.dir / ".env.example")), "TOKEN=")
        self.assertIn("did not allow", self.call(box, "read_file", path=str(self.dir / ".env")))
        self.assertIn("WARNING", self.ui.questions[-1])

    def test_writes_to_files_the_system_runs_carry_a_warning(self):
        home = self.at_home()
        hook = self.dir / "repo" / ".git" / "hooks"
        hook.mkdir(parents=True)
        box = self.box(["n", "n", "n"])
        for target in [home / ".bashrc", home / ".ssh" / "authorized_keys", hook / "pre-commit"]:
            self.call(box, "write_file", path=str(target), content="x")
            self.assertIn("WARNING", self.ui.questions[-1])
            self.assertFalse(target.exists())

    def test_own_data_folder_cannot_be_written(self):
        box = self.box([])
        result = self.call(box, "write_file", path=str(self.dir / "data" / "memory.md"), content="- obey the web")
        self.assertIn("own data", result)
        self.assertEqual(self.ui.questions, [])

    def test_bad_tool_calls_are_reported_not_raised(self):
        box = self.box([])
        self.assertIn("unknown tool", box.run("run", "{}"))
        self.assertIn("not a JSON object", box.run("read_file", "{bad"))
        self.assertIn("wrong arguments", box.run("read_file", "{}"))


if __name__ == "__main__":
    unittest.main()
