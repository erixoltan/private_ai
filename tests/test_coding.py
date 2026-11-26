"""The approval rules, and the coding agent's tools that depend on them."""

import json
import os
import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path

from private_ai.agent import INTERRUPTED, Agent
from private_ai.approvals import Policy, chosen_folder, command_pattern, find_project, split_command
from private_ai.client import ProviderError
from private_ai.providers import Provider
from private_ai.store import Store
from private_ai.tools import Toolbox
from tests.test_agent import FakeModel, FakeUI, HTTPServer


class Case(unittest.TestCase):
    """A project folder and a config folder of its own, so saved rules never leak between tests."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name).resolve()
        self.project = self.dir / "project"
        (self.project / "src").mkdir(parents=True)
        (self.project / "src" / "app.py").write_text("def greet():\n    return 'hello'\n")
        self.rules = self.dir / "config" / "approvals.toml"
        patcher = unittest.mock.patch.dict(os.environ, {"PRIVATE_AI_CONFIG": str(self.dir / "config" / "config.toml")})
        patcher.start()
        self.addCleanup(patcher.stop)

    def policy(self, local=False, trust="ask", project=None):
        return Policy(project or self.project, local, trust)


class PolicyTest(Case):
    def test_with_no_rule_the_user_is_asked(self):
        policy = self.policy()
        for action, target in [("read", str(self.project / "a")), ("write", str(self.project / "a")), ("run", "ls")]:
            self.assertEqual(policy.decide(action, target), ("ask", None))

    def test_an_approval_lasts_as_long_as_its_scope(self):
        inside, other = str(self.project / "src" / "app.py"), self.dir / "other"
        other.mkdir()
        policy = self.policy()
        policy.grant("read", str(self.project), "session")
        self.assertEqual(policy.decide("read", inside)[0], "allow")
        self.assertEqual(self.policy().decide("read", inside)[0], "ask")  # a new session starts with nothing

        policy.grant("write", str(self.project), "project")
        self.assertEqual(self.policy().decide("write", inside)[0], "allow")
        self.assertEqual(self.policy(project=other).decide("write", inside)[0], "ask")

        policy.grant("run", "git status *", "everywhere")
        self.assertEqual(self.policy(project=other).decide("run", "git status --short")[0], "allow")
        self.assertEqual(self.policy(project=other).decide("run", "git push")[0], "ask")

    def test_the_strictest_rule_wins(self):
        self.rules.parent.mkdir()
        self.rules.write_text(
            '[[rule]]\ndecision = "allow"\naction = "run"\nmatch = "*"\n'
            '[[rule]]\ndecision = "deny"\naction = "run"\nmatch = "rm *"\n'
            '[[rule]]\ndecision = "ask"\naction = "run"\nmatch = "git push *"\n'
            '[[rule]]\ndecision = "allow"\naction = "write"\nmatch = "src"\n'
            '[[rule]]\ndecision = "deny"\naction = "write"\nmatch = "src/generated/*"\n'
        )
        policy = self.policy()
        self.assertEqual(policy.decide("run", "make test")[0], "allow")
        self.assertEqual(policy.decide("run", "git push origin main")[0], "ask")
        self.assertEqual(policy.decide("run", "ls && rm -rf build")[0], "deny")  # one forbidden part is enough
        self.assertEqual(policy.decide("write", str(self.project / "src" / "app.py"))[0], "allow")
        self.assertEqual(policy.decide("write", str(self.project / "src" / "generated" / "x.py"))[0], "deny")
        self.assertEqual(policy.decide("write", str(self.project / "README.md"))[0], "ask")

    def test_an_approval_under_a_local_model_does_not_cover_a_cloud_one(self):
        self.policy(local=True).grant("read", str(self.project), "everywhere")
        self.assertIn("local_only = true", self.rules.read_text())
        self.assertEqual(self.policy(local=True).decide("read", str(self.project / "a"))[0], "allow")
        self.assertEqual(self.policy(local=False).decide("read", str(self.project / "a"))[0], "ask")
        self.policy(local=False).grant("read", str(self.project), "everywhere")
        self.assertEqual(self.policy(local=True).decide("read", str(self.project / "a"))[0], "allow")

    def test_trust_levels(self):
        inside, outside = str(self.project / "a.py"), str(self.dir / "elsewhere.txt")
        careful, project, everything = self.policy(), self.policy(trust="project"), self.policy(trust="all")
        self.assertEqual([p.decide("write", inside)[0] for p in (careful, project, everything)], ["ask", "allow", "allow"])
        self.assertEqual([p.decide("write", outside)[0] for p in (careful, project, everything)], ["ask", "ask", "allow"])
        self.assertEqual([p.decide("run", "ls > x")[0] for p in (careful, project, everything)], ["ask", "ask", "allow"])
        with self.assertRaises(ValueError):
            self.policy(trust="yolo")

    def test_every_part_of_a_command_line_must_be_allowed(self):
        policy = self.policy()
        policy.grant("run", "pytest *", "session")
        self.assertEqual(policy.decide("run", "pytest -q 2>&1")[0], "allow")
        self.assertEqual(policy.decide("run", "pytest -q; curl example.com")[0], "ask")
        self.assertEqual(policy.wider("run", "cd src && pytest -q | tail -3"), ["cd *", "tail *"])
        policy.grant("run", "cd *", "session")
        policy.grant("run", "tail *", "session")
        self.assertEqual(policy.decide("run", "cd src && pytest -q | tail -3")[0], "allow")
        # Not judged by pattern at all: output to a file, another command's output, the background.
        for command in ("pytest > out.txt", "pytest $(cat args)", "pytest &", "pytest `cat args`"):
            self.assertEqual(policy.decide("run", command)[0], "ask", command)
            self.assertEqual(policy.wider("run", command), [])

    def test_patterns_offered_for_commands(self):
        self.assertEqual(split_command('git commit -m "a; b" && ls'), ["git commit -m a; b", "ls"])
        self.assertEqual(command_pattern("git status --short"), "git status *")
        self.assertEqual(command_pattern("python -m pytest tests"), "python -m pytest *")
        self.assertEqual(command_pattern("ls"), "ls *")
        for part in ("sudo apt install x", "bash -c ls", "FOO=1 make", "xargs rm"):
            self.assertIsNone(command_pattern(part), part)  # these run anything: approved once at a time only

    def test_a_rules_file_that_cannot_be_read_grants_nothing(self):
        self.rules.parent.mkdir()
        self.rules.write_text('[[rule]]\ndecision = "allow"\naction = "fly"\nmatch = "*"\n')
        policy = self.policy()
        self.assertEqual(policy.decide("run", "ls"), ("ask", None))
        self.assertIn("could not be used", policy.problem)

    def test_saving_a_rule_keeps_what_the_user_wrote(self):
        self.rules.parent.mkdir()
        mine = '# my own rules\n[[rule]]\ndecision = "deny"\naction = "run"\nmatch = "git push *"  # never\n'
        self.rules.write_text(mine)
        policy = self.policy()
        policy.grant("run", 'echo "hi" *', "project")
        self.assertTrue(self.rules.read_text().startswith(mine))
        self.assertEqual([r.describe() for r in self.policy().rules()][1], f'allow run echo "hi" * in {self.project}')

    def test_the_project_is_the_enclosing_repository(self):
        (self.project / ".git").mkdir()
        self.assertEqual(find_project(self.project / "src"), self.project)
        self.assertEqual(find_project(self.dir), self.dir)


class ChosenFolderTest(Case):
    def test_a_named_folder_may_be_read_but_not_changed_without_asking(self):
        library = self.dir / "library"
        library.mkdir()
        policy = self.policy()
        policy.folders = [library]
        self.assertEqual(policy.decide("read", str(library / "lib.py"))[0], "allow")
        self.assertEqual(policy.decide("write", str(library / "lib.py"))[0], "ask")
        self.assertEqual(policy.decide("read", str(self.dir / "elsewhere" / "x"))[0], "ask")
        self.assertEqual(policy.decide("run", "cat " + str(library / "lib.py"))[0], "ask")

    def test_the_users_rules_still_win_inside_a_named_folder(self):
        library = self.dir / "library"
        library.mkdir()
        self.rules.parent.mkdir(parents=True)
        self.rules.write_text(f'[[rule]]\ndecision = "deny"\naction = "read"\nmatch = "{library}/private"\n')
        policy = self.policy()
        policy.folders = [library]
        self.assertEqual(policy.decide("read", str(library / "private" / "notes"))[0], "deny")

    def test_what_can_be_named(self):
        (self.dir / "library").mkdir()
        self.assertEqual(chosen_folder(str(self.dir / "library") + "/"), self.dir / "library")
        self.assertEqual(chosen_folder("library", self.dir), self.dir / "library")
        (self.dir / "config").mkdir()
        for bad in ["", "library", str(self.dir / "missing"), str(self.project / "src" / "app.py"), str(self.dir / "config")]:
            with self.assertRaises(ValueError, msg=bad):
                chosen_folder(bad)

    def test_a_secret_in_a_named_folder_still_warns(self):
        library = self.dir / "library"
        library.mkdir()
        (library / ".env").write_text("KEY=1")
        (library / "lib.py").write_text("x = 1\n")
        ui = FakeUI(["n"])
        provider = Provider(name="cloudy", base_url="https://example.invalid/v1")
        box = Toolbox(Store(self.dir / "data"), ui, provider, True, "ask", self.project, (library,))
        self.assertEqual(box.run("read_file", json.dumps({"path": str(library / "lib.py")})), "x = 1\n")
        self.assertEqual(ui.questions, [])
        self.assertIn("did not allow", box.run("read_file", json.dumps({"path": str(library / ".env")})))
        self.assertIn("WARNING", ui.questions[0])


class CodingToolTest(Case):
    def box(self, answers=(), trust="ask", coding=True, local=False):
        self.ui = FakeUI(list(answers))
        provider = Provider(name="cloudy", base_url="http://127.0.0.1:1/v1" if local else "https://example.invalid/v1")
        return Toolbox(Store(self.dir / "data"), self.ui, provider, coding, trust, self.project)

    def call(self, box, name, **args):
        return box.run(name, json.dumps(args))

    def test_edit_replaces_one_place_and_shows_the_change(self):
        box = self.box(["y"])
        result = self.call(box, "edit_file", path="src/app.py", old_text="'hello'", new_text="'goodbye'")
        self.assertEqual(result, "Edited.")
        self.assertEqual((self.project / "src" / "app.py").read_text(), "def greet():\n    return 'goodbye'\n")
        self.assertIn("-    return 'hello'\n+    return 'goodbye'", self.ui.questions[0])

    def test_edit_needs_text_that_appears_exactly_once(self):
        (self.project / "twice.txt").write_text("a\na\n")
        box = self.box(trust="project")
        self.assertIn("appears 2 times", self.call(box, "edit_file", path="twice.txt", old_text="a", new_text="b"))
        self.assertIn("does not appear", self.call(box, "edit_file", path="twice.txt", old_text="z", new_text="b"))
        self.assertEqual((self.project / "twice.txt").read_text(), "a\na\n")

    def test_one_approval_for_the_project_covers_later_changes(self):
        box = self.box(["p"])
        self.assertEqual(self.call(box, "write_file", path="docs/new/notes.md", content="one"), "Written.")
        self.assertIn(f"changing any file in {self.project}", self.ui.questions[0])
        self.assertEqual(self.call(box, "write_file", path="src/app.py", content="two"), "Written.")
        self.assertEqual(len(self.ui.questions), 1)
        self.assertEqual((self.project / "docs" / "new" / "notes.md").read_text(), "one")  # folders are made as needed
        later = self.box(["n"])  # another session, same project: the saved rule still holds
        self.assertEqual(self.call(later, "write_file", path="src/app.py", content="three"), "Written.")
        self.assertIn("did not allow", self.call(later, "write_file", path=str(self.dir / "outside.txt"), content="x"))

    def test_find_and_search_skip_secrets_and_dependencies(self):
        (self.project / ".env").write_text("TOKEN=hello")
        (self.project / "node_modules").mkdir()
        (self.project / "node_modules" / "lib.py").write_text("hello")
        (self.project / "data.bin").write_bytes(b"\0hello")
        box = self.box(trust="project")
        app = self.project / "src" / "app.py"
        self.assertEqual(self.call(box, "find_files", pattern="*.py"), f"In {self.project}:\n{app}")
        found = self.call(box, "search_files", text="hel+o")
        self.assertEqual(found, f"In {self.project}:\n{app}:2: return 'hello'")
        # Not a valid pattern: taken literally.
        self.assertEqual(self.call(box, "search_files", text="nowhere("), f"In {self.project}:\n(nothing matched)")

    def test_a_coding_task_reads_its_project_without_asking_but_not_other_places(self):
        (self.dir / "other").mkdir()
        (self.dir / "other" / "x.py").write_text("x = 1\n")
        box = self.box(["n", "n"])
        self.assertIn("def greet", self.call(box, "read_file", path="src/app.py"))
        self.assertEqual(self.ui.questions, [])
        self.assertIn("did not allow", self.call(box, "read_file", path=str(self.dir / "other" / "x.py")))
        self.assertIn("did not allow", self.call(box, "write_file", path="src/new.py", content="x"))
        self.assertEqual(len(self.ui.questions), 2)
        # Ordinary chat names no project, so nothing is readable unasked.
        chat = self.box(["n"], coding=False)
        self.assertIn("did not allow", self.call(chat, "read_file", path=str(self.project / "src" / "app.py")))

    def test_a_search_with_no_folder_covers_every_folder_the_task_may_read(self):
        library = self.dir / "library"
        library.mkdir()
        (library / "helper.sh").write_text("echo hello\n")
        box = self.box()
        box.policy.folders = [library]
        found = self.call(box, "find_files", pattern="helper.sh")
        self.assertEqual(found, f"In {self.project}:\n(no file matched)\n\nIn {library}:\n{library / 'helper.sh'}")
        self.assertIn(f"{library / 'helper.sh'}:1: echo hello", self.call(box, "search_files", text="echo"))
        self.assertEqual(self.ui.questions, [])
        self.assertIn("does not exist", self.call(box, "find_files", pattern="*", path=str(self.dir / "nowhere")))

    def test_a_search_gives_up_on_a_folder_that_is_too_big_and_can_be_stopped(self):
        for n in range(30):
            (self.project / f"f{n:02}.txt").write_text("needle\n")
        box = self.box()
        with unittest.mock.patch("private_ai.tools.MAX_WALKED", 10):
            found = self.call(box, "find_files", pattern="*.txt")
            self.assertIn("NOT COMPLETE", found)
            self.assertIn("stopped after 10 files", found)
            self.assertEqual(found.count(".txt"), 10)
            self.assertIn("NOT COMPLETE", self.call(box, "search_files", text="needle"))
        with unittest.mock.patch("private_ai.tools.MAX_WALK_SECONDS", -1):
            self.assertIn("stopped after 0 files", self.call(box, "find_files", pattern="*.txt"))
        self.assertNotIn("NOT COMPLETE", self.call(box, "find_files", pattern="*.txt"))
        box.stopped = lambda: True
        self.assertEqual(self.call(box, "find_files", pattern="*.txt"), "Error: stopped by the user.")

    def test_command_runs_in_the_project_and_reports_output_and_exit_code(self):
        box = self.box(["y", "y"])
        self.assertEqual(self.call(box, "run_command", command="pwd"), f"exit code 0\n{self.project}")
        self.assertIn(f"$ pwd\nThe output will be sent to cloudy (cloud).", self.ui.questions[0])
        self.assertEqual(self.call(box, "run_command", command="echo oops >&2; exit 3"), "exit code 3\noops")
        self.assertEqual(len(self.ui.questions), 2)

    def test_a_program_that_asks_questions_is_given_its_answers(self):
        (self.project / "ask.py").write_text("a = input('name? ')\nb = input('again? ')\nprint('got', a, b)\ninput()\n")
        box = self.box(["y"], trust="project")
        result = self.call(box, "run_command", command="python3 ask.py", input="Sally\nyes")
        self.assertIn("got Sally yes", result)
        self.assertIn("EOFError", result)  # past the answers given, it reads end-of-input
        self.assertIn("Typed into it:\nSally\nyes", self.ui.questions[0])
        box = self.box(trust="all")
        self.assertEqual(self.call(box, "run_command", command="true", input="never read"), "exit code 0\n(no output)")

    def test_an_edit_forgives_spacing_and_says_what_is_there_otherwise(self):
        app = self.project / "src" / "app.py"
        app.write_text("def pause():\n    try:\n        go()\n    except EOFError:\n        pass  # carry on\n\nx = 1\n")
        box = self.box(trust="project")
        old = "def pause():\n    try:\n        go()\n    except EOFError:\n        pass   # carry on"  # one space too many
        self.assertIn("spacing only", self.call(box, "edit_file", path="src/app.py", old_text=old, new_text="def pause():\n    pass"))
        self.assertEqual(app.read_text(), "def pause():\n    pass\n\nx = 1\n")
        missed = self.call(box, "edit_file", path="src/app.py", old_text="def pause():\n    return", new_text="")
        self.assertIn("The closest text, at line 1, is shown below", missed)
        self.assertTrue(missed.endswith("def pause():\n    pass"))
        self.assertIn("Nothing like its first line", self.call(box, "edit_file", path="src/app.py", old_text="zzzz qqqq", new_text=""))
        app.write_text("a  = 1\na = 1\n")
        self.assertIn("appears 2 times, apart from spacing", self.call(box, "edit_file", path="src/app.py", old_text="a   = 1", new_text=""))

    def test_a_path_with_stray_spaces_gets_a_hint(self):
        box = self.box(trust="project")
        spaced = str(self.project).replace("/project", "/ project") + "/src/ app.py"
        for tool, args in [("read_file", {}), ("edit_file", {"old_text": "a", "new_text": "b"})]:
            self.assertIn(f"Did you mean {self.project / 'src' / 'app.py'}?", self.call(box, tool, path=spaced, **args))
        self.assertIn("Use write_file", self.call(box, "edit_file", path="src/none.py", old_text="a", new_text="b"))

    def test_command_approval_can_be_widened_by_kind(self):
        box = self.box(["s", "n"])
        self.call(box, "run_command", command="echo one")
        self.assertIn("commands of this kind: echo *", self.ui.questions[0])
        self.assertEqual(self.call(box, "run_command", command="echo two"), "exit code 0\ntwo")
        self.assertIn("did not allow", self.call(box, "run_command", command="echo three; touch made"))
        self.assertFalse((self.project / "made").exists())
        self.assertEqual(len(self.ui.questions), 2)

    def test_a_command_that_runs_anything_is_only_ever_approved_once(self):
        box = self.box(["n"])
        self.call(box, "run_command", command="bash -c 'echo hi'")
        self.assertNotIn("wider approval", self.ui.questions[0])

    def test_a_forbidden_command_is_refused_without_asking(self):
        self.rules.parent.mkdir()
        self.rules.write_text('[[rule]]\ndecision = "deny"\naction = "run"\nmatch = "git push *"\n')
        box = self.box(trust="all")
        self.assertIn("rules forbid this", self.call(box, "run_command", command="git push origin main"))
        self.assertEqual(self.ui.questions, [])

    def test_a_command_naming_secrets_is_asked_about_whatever_the_rules_say(self):
        box = self.box(["s", "n"])
        self.call(box, "run_command", command="cat src/app.py")
        result = self.call(box, "run_command", command="cat ~/.ssh/id_ed25519")
        self.assertIn("did not allow", result)
        self.assertIn("WARNING", self.ui.questions[1])
        self.assertNotIn("wider approval", self.ui.questions[1])

    def test_trust_all_asks_nothing(self):
        box = self.box(trust="all")
        self.assertEqual(self.call(box, "write_file", path=".env", content="A=1"), "Written.")
        self.assertEqual(self.call(box, "run_command", command="cat .env > copy; cat copy"), "exit code 0\nA=1")
        self.assertEqual(self.ui.questions, [])

    def test_commands_do_not_get_this_programs_keys(self):
        with unittest.mock.patch.dict(os.environ, {"FIREWORKS_API_KEY": "sk-secret", "OTHER_SETTING": "kept"}):
            box = self.box(trust="all")
            result = self.call(box, "run_command", command="echo [$FIREWORKS_API_KEY] [$OTHER_SETTING]")
        self.assertEqual(result, "exit code 0\n[] [kept]")

    def test_command_is_stopped_at_its_time_limit_with_everything_it_started(self):
        box = self.box(trust="all")
        started = time.monotonic()
        result = self.call(box, "run_command", command="echo begun; sleep 30 | sleep 30", seconds="1")
        self.assertLess(time.monotonic() - started, 10)
        self.assertTrue(result.startswith("stopped at the time limit of 1 seconds. Nothing after that point ran."), result)
        self.assertTrue(result.endswith("\nbegun"), result)

    def test_what_a_python_program_printed_is_kept_when_it_is_stopped(self):
        box = self.box(trust="all")
        result = self.call(box, "run_command", command="python3 -c 'import time; print(\"so far\"); time.sleep(30)'", seconds="1")
        self.assertTrue(result.endswith("\nso far"), result)

    def test_any_command_in_this_task_can_be_approved_at_once(self):
        box = self.box(["a"])
        heredoc = "python3 - <<'EOF'\nprint('ran')\nEOF"  # no pattern could cover this
        self.assertEqual(self.call(box, "run_command", command=heredoc), "exit code 0\nran")
        self.assertIn("any command in this task", self.ui.questions[0].lower())
        self.assertEqual(self.call(box, "run_command", command="echo again > /dev/null; echo x"), "exit code 0\nx")
        self.assertEqual(len(self.ui.questions), 1)
        self.assertTrue(box.policy.free_run)
        self.assertFalse(box.policy.path.exists())  # for this session only: nothing was written to the rules
        box.policy.set_free_run(False)
        self.ui.answers.append("n")
        self.assertIn("Error", self.call(box, "run_command", command="echo more"))

    def test_any_command_still_gives_way_to_ask_and_deny_rules_and_to_warnings(self):
        box = self.box(["a", "n", "n"])
        box.policy.path.parent.mkdir(parents=True, exist_ok=True)
        box.policy.path.write_text(
            '[[rule]]\ndecision = "ask"\naction = "run"\nmatch = "git push *"\n'
            '[[rule]]\ndecision = "deny"\naction = "run"\nmatch = "rm *"\n'
        )
        self.call(box, "run_command", command="echo hi")
        self.assertIn("Error", self.call(box, "run_command", command="git push origin main"))
        self.assertNotIn("any command", self.ui.questions[1])
        self.assertIn("forbid", self.call(box, "run_command", command="rm -rf x"))
        self.assertIn("Error", self.call(box, "run_command", command=f"cat {box.policy.path}"))
        self.assertIn("WARNING", self.ui.questions[2])

    def test_command_is_stopped_when_the_turn_is(self):
        box = self.box(trust="all")
        threading.Timer(0.3, lambda: setattr(box, "stopped", lambda: True)).start()
        started = time.monotonic()
        self.assertIn("stopped by the user", self.call(box, "run_command", command="sleep 30"))
        self.assertLess(time.monotonic() - started, 10)

    def test_long_output_keeps_its_start_and_end(self):
        box = self.box(trust="all")
        result = self.call(box, "run_command", command="echo FIRST; seq 1 20000; echo LAST")
        self.assertIn("FIRST", result)
        self.assertTrue(result.endswith("LAST"))
        self.assertIn("bytes of output not shown", result)
        self.assertLess(len(result), 20_000)

    def test_ordinary_chat_has_no_coding_tools(self):
        box = self.box(trust="all", coding=False)
        self.assertIn("unknown tool", self.call(box, "run_command", command="ls"))
        self.assertIn("unknown tool", self.call(box, "edit_file", path="a", old_text="b", new_text="c"))


class CodingAgentTest(Case):
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
        super().setUp()
        self.store = Store(self.dir / "data")
        FakeModel.script, FakeModel.requests = [], []
        FakeModel.side_replies, FakeModel.side_requests = [], []
        here = os.getcwd()
        os.chdir(self.project)
        self.addCleanup(os.chdir, here)

    def test_a_coding_task_edits_and_runs_in_its_project(self):
        FakeModel.script = [
            ("edit_file", {"path": "src/app.py", "old_text": "'hello'", "new_text": "'hi'"}),
            ("run_command", {"command": "python3 -c 'import sys; sys.path.insert(0, \"src\"); import app; print(app.greet())'"}),
            "Changed the greeting and checked it.",
        ]
        ui = FakeUI(answers=["y", "y"])
        agent = Agent(self.provider, "m", self.store, ui, coding=True)
        agent.send("make it say hi")
        request = FakeModel.requests[-1]
        self.assertIn("run_command", [t["function"]["name"] for t in request["tools"]])
        self.assertIn(f"coding agent in the project at {self.project}", request["messages"][0]["content"])
        self.assertEqual(request["messages"][-1]["content"], "exit code 0\nhi")
        saved = self.store.load(agent.conversation["id"])
        self.assertEqual((saved["mode"], saved["project"]), ("code", str(self.project)))
        # Continued later from another folder, it is still a coding task in the same project.
        os.chdir(self.dir)
        again = Agent(self.provider, "m", self.store, FakeUI(), saved)
        self.assertTrue(again.coding)
        self.assertEqual(again.toolbox.cwd, self.project)

    def test_a_task_works_in_the_folder_it_is_given_and_reads_the_others(self):
        library = self.dir / "library"
        library.mkdir()
        (library / "lib.py").write_text("def helper(): pass\n")
        os.chdir(self.dir)  # started somewhere else entirely
        FakeModel.script = [
            ("read_file", {"path": "src/app.py"}),
            ("read_file", {"path": str(library / "lib.py")}),
            ("read_file", {"path": str(self.dir / "config" / "elsewhere.txt")}),
            ("read_file", {"path": str(self.dir / "stray.txt")}),
            "Read both.",
            "And again.",
        ]
        (self.dir / "stray.txt").write_text("stray\n")
        ui = FakeUI(answers=["y"])
        agent = Agent(self.provider, "m", self.store, ui, coding=True, project=self.project, folders=(library,))
        agent.send("look at the app and the library")
        self.assertEqual(len(ui.questions), 1)  # only the file outside the folders it was given
        self.assertIn(str(self.dir / "stray.txt"), ui.questions[0])
        prompt = FakeModel.requests[-1]["messages"][0]["content"]
        self.assertIn(f"coding agent in the project at {self.project}", prompt)
        self.assertIn(f"- {library}", prompt)
        # Both are kept with the conversation, and a change to the list takes effect on the next message.
        saved = self.store.load(agent.conversation["id"])
        self.assertEqual((saved["project"], saved["folders"]), (str(self.project), [str(library)]))
        saved["folders"] = []
        again = Agent(self.provider, "m", self.store, FakeUI(), saved)
        again.send("again")
        self.assertEqual(again.toolbox.policy.folders, [])
        self.assertNotIn(str(library), FakeModel.requests[-1]["messages"][0]["content"])

    def test_a_tool_call_written_as_text_is_sent_back_rather_than_taken_as_the_answer(self):
        FakeModel.script = [
            "<tool_call>\n<function=read_File> path=src/ app.py\n</function>",
            ("read_file", {"path": "src/app.py"}),
            "It defines greet.",
            "<tool_call>one", "<tool_call>two", "<tool_call>three",
        ]
        ui = FakeUI()
        agent = Agent(self.provider, "m", self.store, ui, coding=True)
        agent.send("what is in the app?")
        self.assertIn("automatic note", FakeModel.requests[1]["messages"][-1]["content"].lower())
        self.assertIn("<tool_call>", FakeModel.requests[1]["messages"][-2]["content"])
        self.assertTrue(any("tool call as text" in n for n in ui.notices))
        # Neither the garbled reply nor the note is kept.
        saved = self.store.load(agent.conversation["id"])["messages"]
        self.assertEqual([m["role"] for m in saved], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(saved[-1]["content"], "It defines greet.")
        # It is not asked for ever: after two tries the reply stands.
        agent.send("and again?")
        self.assertEqual(self.store.load(agent.conversation["id"])["messages"][-1]["content"], "<tool_call>three")

    def test_an_empty_reply_is_sent_back_rather_than_ending_the_work_in_silence(self):
        FakeModel.script = ["", ("read_file", {"path": "src/app.py"}), "It defines greet.", "", " ", "\n"]
        ui = FakeUI()
        agent = Agent(self.provider, "m", self.store, ui, coding=True)
        agent.send("what is in the app?")
        self.assertIn("was empty", FakeModel.requests[1]["messages"][-1]["content"])
        saved = self.store.load(agent.conversation["id"])["messages"]
        self.assertEqual([m["role"] for m in saved], ["user", "assistant", "tool", "assistant"])
        agent.send("and again?")  # a model that stays silent is reported, not waited on for ever
        self.assertIn("empty reply again", ui.notices[-1])

    def test_work_done_before_a_failure_is_kept(self):
        FakeModel.script = [("remember", {"fact": "Likes plums."}), 500]
        agent = Agent(self.provider, "m", self.store, FakeUI())
        with self.assertRaises(ProviderError):
            agent.send("remember that I like plums, then tell me a story")
        self.assertTrue(agent.partial)
        saved = self.store.load(agent.conversation["id"])["messages"]
        self.assertEqual([m["role"] for m in saved], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(saved[-1]["content"], INTERRUPTED)
        FakeModel.script = ["Here is the story."]
        agent.send("go on")  # the history is one a model accepts
        self.assertFalse(agent.partial)

    def test_a_message_that_fails_at_once_is_not_kept(self):
        FakeModel.script = [500]
        agent = Agent(self.provider, "m", self.store, FakeUI())
        with self.assertRaises(ProviderError):
            agent.send("hello")
        self.assertEqual((agent.partial, agent.conversation["messages"]), (False, []))

    def test_a_conversation_cut_off_by_the_program_ending_is_closed_on_loading(self):
        conversation = self.store.new_conversation()
        conversation["messages"] = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "remember", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Saved."},
        ]
        agent = Agent(self.provider, "m", self.store, FakeUI(), conversation)
        self.assertEqual(agent.conversation["messages"][-1], {"role": "assistant", "content": INTERRUPTED})


if __name__ == "__main__":
    unittest.main()
