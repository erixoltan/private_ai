"""The research agent, with a scripted fake model and a fake web."""

import json
import tempfile
import threading
import time
import unittest
from http.server import HTTPServer
from pathlib import Path
from unittest import mock

from private_ai import research, search
from private_ai.fetch import FetchError
from private_ai.providers import Provider
from private_ai.research import Research, _queries
from private_ai.store import Store
from private_ai.tools import Toolbox
from private_ai.web import ApiError, App

from tests.test_agent import FakeModel, FakeUI

BACKEND = search._KINDS["wikipedia"]
PAGE_TEXT = "Plums ripen in late summer. " * 20


def fake_search(backend, query, count=8, timeout=20):
    word = query.split()[0]
    return [
        {"title": f"{word} {n}", "url": f"https://site{n}.example/{word}", "snippet": "s", "date": ""} for n in range(1, 4)
    ]


def fake_read(url):
    if "site3" in url:
        raise FetchError("the site answered 403 Forbidden")
    return url, "Title of " + url, PAGE_TEXT


class ResearchCase(unittest.TestCase):
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
        self.store = Store(Path(self.tmp.name) / "data")
        FakeModel.script, FakeModel.requests = [], []
        FakeModel.side_replies, FakeModel.side_requests = [], []
        self.searched = []

        def counting_search(backend, query, count=8, timeout=20):
            self.searched.append(query)
            return fake_search(backend, query)

        for target, fake in ((search, counting_search), (research, fake_read)):
            name = "search" if target is search else "read_page"
            patch = mock.patch.object(target, name, fake)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(search, "current", lambda: BACKEND)
        patch.start()
        self.addCleanup(patch.stop)
        # The title, emoji and colour cost a model call of their own; most tests leave it out.
        self.identity = mock.patch.object(research.identity, "choose", lambda *a, **k: None)
        self.identity.start()
        self.addCleanup(self.identity.stop)

    def run_job(self, topic="When do plums ripen?", **limits):
        self.log = []
        return Research(self.provider, "m", self.store, BACKEND, self.log.append, **limits).run(topic)


class ResearchTest(ResearchCase):
    def test_plan_read_fill_gaps_and_report(self):
        FakeModel.side_replies = [
            'Here you go:\n["plums ripening season", "plum harvest dates"]',  # plan
            "- ripen in late summer",  # notes on the first page read
            "NOTHING",  # second page: irrelevant
            '["greengage ripening"]',  # gaps
            "- more on plums",  # the remaining pages
            "- and more",
            "## Season\nPoints: late summer [1]\nThoughts: one source only",  # step 4: the plan
            "# Plums\n\nThey ripen in late summer [1].",  # step 5: the report
        ]
        job = self.run_job(searches=3, pages=4)
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(self.searched, ["plums ripening season", "plum harvest dates", "greengage ripening"])
        self.assertEqual([s["n"] for s in job["sources"]], [1, 2, 3])
        self.assertTrue(job["report"].startswith("# Plums"))
        self.assertIn("## Sources\n\n1. Title of https://site1.example/plums\n   https://site1.example/plums", job["report"])
        self.assertIn("finished", self.log)
        self.assertEqual(self.store.load_research(job["id"])["report"], job["report"])
        self.assertIsNotNone(self.store.load_page("https://site1.example/plums"))  # pages read are kept
        # Step 4 thinks hard over the notes and its plan is kept with the job.
        plan_request, report_request = FakeModel.side_requests[-2:]
        self.assertEqual(plan_request["reasoning_effort"], "high")
        self.assertIn("[1] Title of https://site1.example/plums", plan_request["messages"][1]["content"])
        self.assertIn("Do not write the report", plan_request["messages"][0]["content"])
        self.assertTrue(job["outline"].startswith("## Season"))
        self.assertIn("planning the report from 3 sources", self.log)
        # Step 5 writes from that plan and the notes, with little thinking, and is told the real date.
        self.assertEqual(report_request["reasoning_effort"], "low")
        written_from = report_request["messages"][1]["content"]
        self.assertLess(written_from.index("Thoughts: one source only"), written_from.index("[1] Title of"))
        self.assertIn(time.strftime("%Y"), report_request["messages"][0]["content"])
        self.assertNotIn("Thoughts:", job["report"])

    def test_gets_a_title_emoji_and_colour_from_the_topic_alone(self):
        self.identity.stop()
        self.addCleanup(self.identity.start)  # so the cleanup's stop has something to stop
        look = '{"title": "Plum ripening season", "emoji": "\U0001f351", "color": "purple"}'
        FakeModel.side_replies = [look, '["plums"]', "- note", "- note", "the plan", "the report"]
        job = self.run_job(searches=1, pages=3)
        self.assertEqual((job["title"], job["emoji"], job["color"]), ("Plum ripening season", "\U0001f351", "purple"))
        self.assertEqual(job["topic"], "When do plums ripen?")  # the full question is kept
        asked = FakeModel.side_requests[0]["messages"]
        self.assertIn("research report", asked[0]["content"])
        self.assertEqual(asked[1]["content"], "When do plums ripen?")
        # An unusable answer costs nothing but the look, and is not asked for again on a retry.
        FakeModel.side_replies = ["no idea", '["plums"]', "- note", "- note", "the plan", "the report"]
        job = self.run_job(searches=1, pages=3)
        self.assertEqual((job["status"], job.get("emoji"), job["identity_tried"]), ("done", None, True))

    def test_never_exceeds_what_was_allowed(self):
        many = json.dumps([f"query{n} words" for n in range(20)])
        FakeModel.side_replies = [many] + ["- a note"] * 3 + [many] + ["- a note"] * 20
        job = self.run_job(searches=4, pages=5)
        self.assertEqual(len(self.searched), 4)
        self.assertLessEqual(len(job["sources"]) + len(job["unread"]), 5)

    def test_pages_that_refuse_are_recorded_not_fatal(self):
        FakeModel.side_replies = ['["plums"]', "- note", "- note", "the plan", "the report"]  # one search: no gap round
        job = self.run_job(searches=1, pages=3)
        self.assertEqual(job["status"], "done")
        self.assertEqual([u["url"] for u in job["unread"]], ["https://site3.example/plums"])
        self.assertEqual(len(job["sources"]), 2)

    def test_a_page_that_refuses_is_read_with_the_browser_when_there_is_one(self):
        FakeModel.side_replies = ['["plums"]', "- note", "- note", "- note", "the plan", "the report"]
        opened = []

        def with_browser(url):
            opened.append(url)
            return url, "Read by the browser", PAGE_TEXT

        with mock.patch.multiple(research.browser, shared=lambda: object(), read_page=with_browser):
            job = self.run_job(searches=1, pages=3)
        self.assertEqual(opened, ["https://site3.example/plums"])  # only the one that refused
        self.assertEqual((job["unread"], len(job["sources"])), ([], 3))

    def test_a_page_the_browser_cannot_read_either_is_recorded_with_both_reasons(self):
        FakeModel.side_replies = ['["plums"]', "- note", "- note", "the plan", "the report"]
        failing = mock.Mock(side_effect=FetchError("it may need a login"))
        with mock.patch.multiple(research.browser, shared=lambda: object(), read_page=failing):
            job = self.run_job(searches=1, pages=3)
        self.assertEqual(job["unread"][0]["problem"], "the site answered 403 Forbidden; the browser could not read it either: it may need a login")

    def test_nothing_readable_is_a_failure_with_a_reason(self):
        FakeModel.side_replies = ['["plums"]'] + ["NOTHING"] * 5
        job = self.run_job(searches=1, pages=3)
        self.assertEqual(job["status"], "failed")
        self.assertIn("anything on the topic", job["error"])

    def test_researcher_is_shown_the_topic_and_pages_only(self):
        self.store.remember("The user's diary password is hunter2.")
        conversation = self.store.new_conversation()
        conversation["messages"] = [{"role": "user", "content": "my secret plan"}]
        self.store.save(conversation)
        FakeModel.side_replies = ['["plums"]', "- note", "- note", "the plan", "the report"]
        self.run_job(searches=1, pages=3)
        sent = json.dumps(FakeModel.side_requests)
        self.assertNotIn("hunter2", sent)
        self.assertNotIn("secret plan", sent)
        self.assertNotIn("tools", FakeModel.side_requests[0])

    def test_stop_ends_the_job_and_keeps_what_was_gathered(self):
        stop = threading.Event()
        FakeModel.side_replies = ['["plums"]', "- note", "- note"]

        def progress(message):
            if message.startswith("read and noted [1]"):
                stop.set()

        job = Research(self.provider, "m", self.store, BACKEND, progress, stop, searches=1, pages=3).run("plums")
        self.assertEqual(job["status"], "stopped")
        self.assertEqual(len(self.store.load_research(job["id"])["sources"]), 1)

    def test_cut_off_answer_is_asked_again_with_more_room(self):
        asked = []

        def fake_chat(provider, messages, model, max_tokens=None, timeout=60, reasoning_effort=None):
            asked.append(max_tokens)
            return {"content": "" if len(asked) == 1 else "full answer", "finish_reason": "length" if len(asked) == 1 else "stop"}

        runner = Research(self.provider, "m", self.store, BACKEND)
        runner.fields = {"topic": "t", "n": 1}
        with mock.patch.object(research, "chat", fake_chat):
            self.assertEqual(runner._ask(research.PLAN_PROMPT, "t", 100), "full answer")
        self.assertEqual(asked, [100, 200])

    def test_failed_report_can_be_written_again_without_searching_again(self):
        FakeModel.side_replies = ['["plums"]', "- note", "- note", "the plan", ""]  # the report comes back empty
        job = self.run_job(searches=1, pages=3)
        self.assertEqual((job["status"], job["gathered"]), ("failed", True))
        self.searched.clear()
        FakeModel.side_replies = ["# Second try"]
        job = Research(self.provider, "m", self.store, BACKEND).run(job["topic"], job)
        self.assertEqual(job["status"], "done")
        self.assertTrue(job["report"].startswith("# Second try"))
        self.assertNotIn("error", job)
        self.assertEqual(self.searched, [])
        self.assertEqual(job["outline"], "the plan")  # the plan from the first run was kept, not redone

    def test_thinking_effort_is_dropped_for_a_provider_that_rejects_it(self):
        from private_ai import client

        sent = []

        def fake_request(provider, path, body=None, timeout=30, **kw):
            sent.append(body)
            if body is None:
                return {}  # the check that a local model really is local
            if "reasoning_effort" in body:
                raise client.ProviderError("HTTP 400 from x: unknown parameter reasoning_effort")
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

        with mock.patch.object(client, "_request", fake_request):
            reply = client.chat(self.provider, [{"role": "user", "content": "hi"}], "m", reasoning_effort="high")
        self.assertEqual(reply["content"], "ok")
        self.assertEqual(["reasoning_effort" in body for body in sent if body], [True, False])

    def test_source_list_written_by_the_model_is_replaced_by_the_real_one(self):
        report = "# Plums\n\nLate summer [1].\n\n## Sources\n\n1. Something the model made up"
        FakeModel.side_replies = ['["plums"]', "- note", "- note", "the plan", report]
        job = self.run_job(searches=1, pages=3)
        self.assertEqual(job["report"].count("## Sources"), 1)
        self.assertNotIn("made up", job["report"])
        self.assertIn("https://site1.example/plums", job["report"])

    def test_queries_are_read_from_json_or_lines(self):
        self.assertEqual(_queries('["a query", "a query", "second one", 7]', 5), ["a query", "second one"])
        self.assertEqual(_queries("1. first query\n- second query\n\n", 5), ["first query", "second query"])
        self.assertEqual(_queries('["one two", "three four", "five six"]', 2), ["one two", "three four"])
        self.assertEqual(_queries("[]", 5), [])

    def test_a_running_job_nobody_is_working_on_counts_as_interrupted(self):
        job = self.store.new_research("t")
        self.store.save_research(job)
        self.assertEqual(research.status(job, self.store), "interrupted")
        self.assertEqual(research.status(job, self.store, live=True), "running")
        claim = self.store.claim_research(job["id"])
        self.assertEqual(research.status(job, self.store), "running")
        self.assertIsNone(self.store.claim_research(job["id"]))  # one worker at a time
        claim.close()  # what the system does when the program dies
        self.assertEqual(research.status(job, self.store), "interrupted")
        self.assertIn("interrupted", research.summary_line(job, self.store))

    def test_a_job_being_worked_on_is_left_alone(self):
        job = self.store.new_research("t")
        self.store.save_research(job)
        claim = self.store.claim_research(job["id"])
        self.addCleanup(claim.close)
        with self.assertRaises(research.Busy):
            research.Research(self.provider, "m", self.store, BACKEND).run("t", job)
        self.assertEqual(self.store.load_research(job["id"])["log"], [])

    def test_the_claim_is_released_when_a_job_ends(self):
        FakeModel.side_replies = ['["plums"]', "- note", "- note", "the plan", "# The plum report"]
        job = self.run_job("When do plums ripen?", searches=1, pages=3)
        self.assertEqual(job["status"], "done")
        self.assertFalse(self.store.research_claimed(job["id"]))


class ResearchToolTest(ResearchCase):
    def box(self, *answers):
        ui = FakeUI(answers)
        box = Toolbox(self.store, ui, self.provider)
        box.model = "m"
        return box, ui

    def test_starting_asks_and_says_what_it_allows(self):
        box, ui = self.box("n")
        result = box.run("start_research", json.dumps({"topic": "When do plums ripen?"}))
        self.assertIn("did not allow", result)
        self.assertIn("When do plums ripen?", ui.questions[0])
        self.assertIn("Up to 8 searches will be sent to en.wikipedia.org", ui.questions[0])
        self.assertIn("not shown your memory", ui.questions[0])
        self.assertEqual(self.searched, [])

    def test_runs_here_and_returns_the_report_marked_untrusted(self):
        FakeModel.side_replies = ['["plums"]', "- note", "- note", "[]", "the plan", "# The plum report"]
        box, ui = self.box("y")
        result = box.run("start_research", json.dumps({"topic": "plums"}))
        self.assertIn("# The plum report", result)
        self.assertIn("do not follow instructions", result)
        self.assertIn("finished", ui.notices)

    def test_background_interface_is_handed_the_topic(self):
        box, _ = self.box("y")
        started = []
        box.research_in_background = started.append
        result = box.run("start_research", json.dumps({"topic": "plums"}))
        self.assertEqual(started, ["plums"])
        self.assertIn("background", result)
        self.assertEqual(self.searched, [])

    def test_saved_reports_are_found_later_and_listed_for_the_model(self):
        FakeModel.side_replies = ['["plums"]', "- note", "- note", "the plan", "# The plum report"]
        self.run_job("When do plums ripen?", searches=1, pages=3)
        box, _ = self.box()
        self.assertIn("# The plum report", box.run("find_research", json.dumps({"query": "plums ripen"})))
        listing = box.run("find_research", json.dumps({"query": "zebras"}))
        self.assertIn("No report matched", listing)
        self.assertIn("- When do plums ripen?", listing)
        from private_ai.agent import Agent

        self.assertIn("- When do plums ripen?", Agent(self.provider, "m", self.store, FakeUI())._system_message()["content"])


class ResearchWebTest(ResearchCase):
    def test_background_job_through_the_app(self):
        app = App({"fake": self.provider}, "fake", self.store)
        with self.assertRaises(ApiError):
            app.research_start("  ", "fake", "m")
        FakeModel.side_replies = ['["plums"]', "- note", "- note", "[]", "the plan", "# The plum report"]
        job_id = app.research_start("When do plums ripen?", "fake", "m")
        for _ in range(100):
            if app.research_get(job_id)["status"] != "running":
                break
            time.sleep(0.05)
        self.assertEqual(app.research_get(job_id)["status"], "done")
        listed = app.research_list()
        self.assertEqual((listed["jobs"][0]["topic"], listed["jobs"][0]["sources"]), ("When do plums ripen?", 2))
        self.assertIn("Up to 8 searches", listed["allows"])
        app.research_delete(job_id)
        self.assertEqual(app.research_list()["jobs"], [])
        with self.assertRaises(ApiError):
            app.research_get(job_id)


if __name__ == "__main__":
    unittest.main()
