"""The browser, tested against a stand-in for Firefox: no real browser is started and nothing leaves this machine."""

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from private_ai import browser, research, tools
from private_ai.fetch import FetchError
from private_ai.providers import Provider
from private_ai.store import Store
from tests.test_agent import FakeUI

PAGE = "http://127.0.0.1:9/page"  # never contacted: the stand-in answers for every address


def stand_in(test, tmp):
    """A browser that is really tests/fake_firefox.py, and name lookups that stay on this machine."""
    program = Path(tmp) / "firefox"
    fake = Path(__file__).parent / "fake_firefox.py"
    program.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
    program.chmod(program.stat().st_mode | stat.S_IXUSR)

    def public(host, port, allow_private=False):
        if host.startswith("127."):
            raise FetchError("only public addresses can be opened")

    patch = mock.patch.object(browser, "_public_address", public)
    patch.start()
    test.addCleanup(patch.stop)
    found = browser.Browser(str(program), Path(tmp) / "profile", idle_seconds=0, settle_seconds=1)
    test.addCleanup(found.close)
    return found


class BrowserTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.program = self.dir / "firefox"
        fake = Path(__file__).parent / "fake_firefox.py"
        self.program.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
        self.program.chmod(self.program.stat().st_mode | stat.S_IXUSR)
        self.profile = self.dir / "profile"
        self.browser = browser.Browser(str(self.program), self.profile, idle_seconds=0, settle_seconds=2)
        self.addCleanup(self.browser.close)

    def opened(self):
        return (self.profile / "opened").read_text().split()

    def test_reads_a_page_then_leaves_it_and_quits_when_closed(self):
        url, title, text = self.browser.read(PAGE, allow_private=True)
        self.assertEqual((url, title), (PAGE, "Built"))
        self.assertIn("the page as the browser built it", text)
        self.assertEqual(self.opened(), [PAGE, "about:blank"])  # the page is not left running
        self.assertIn("--headless", (self.profile / "started").read_text())
        process = self.browser.process
        self.browser.read(PAGE + "?again", allow_private=True)
        self.assertIs(self.browser.process, process)  # one browser, kept for the next page
        self.browser.close()
        self.assertEqual(process.poll(), 0)
        self.assertIsNone(self.browser.process)

    def test_profile_is_set_up_to_keep_to_itself_and_off_the_local_network(self):
        self.browser.read(PAGE, allow_private=True)
        prefs = (self.profile / "user.js").read_text()
        self.assertIn('"network.proxy.type", 2', prefs)
        self.assertIn("data:application/x-ns-proxy-autoconfig;base64,", prefs)
        self.assertIn('"signon.rememberSignons", false', prefs)
        self.assertIn('"toolkit.telemetry.enabled", false', prefs)
        self.assertIn(str(self.profile / "downloads"), prefs)
        self.assertEqual(stat.S_IMODE(self.profile.stat().st_mode), 0o700)

    def test_local_addresses_are_refused_before_the_browser_is_even_started(self):
        for url in ("http://127.0.0.1:9/x", "http://localhost/x", "http://192.168.1.1/", "file:///etc/passwd", "about:config"):
            with self.assertRaises(FetchError):
                self.browser.read(url)
        self.assertIsNone(self.browser.process)

    def test_a_page_that_ends_up_on_this_machine_is_not_read(self):
        with mock.patch.object(browser, "_check_public", side_effect=[None, browser.BrowserError("local network")]) as check:
            with self.assertRaisesRegex(browser.BrowserError, "local network"):
                self.browser.read("http://example.org/wanders")
        self.assertEqual(check.call_args_list[1].args[0], "http://127.0.0.1:9/inside")

    def test_a_page_with_nothing_to_read_is_an_error(self):
        with self.assertRaisesRegex(browser.BrowserError, "almost no readable text"):
            self.browser.read(PAGE + "/thin", allow_private=True)

    def test_a_captcha_is_reported_and_left_for_the_user(self):
        with self.assertRaisesRegex(browser.BrowserError, "proof that a person is there.*private-ai browser http"):
            self.browser.read(PAGE + "/guarded", allow_private=True)

    def test_a_page_too_slow_to_finish_is_read_as_far_as_it_got(self):
        self.assertIn("as the browser built it", self.browser.read(PAGE + "/slow", allow_private=True)[2])

    def test_text_only_drawn_on_screen_is_still_read(self):
        _, title, text = self.browser.read(PAGE + "/drawn", allow_private=True)
        self.assertEqual(title, "Built")
        self.assertIn("as the browser built it", text)

    def test_a_browser_that_will_not_start_says_so(self):
        self.program.write_text("#!/bin/sh\nexit 1\n")
        with self.assertRaisesRegex(browser.BrowserError, "would not start"):
            self.browser.read(PAGE, allow_private=True)
        self.assertIsNone(self.browser.process)

    def test_a_browser_that_dies_is_started_again_for_the_next_page(self):
        self.browser.read(PAGE, allow_private=True)
        self.browser.process.kill()
        self.browser.process.wait()
        self.assertIn("as the browser built it", self.browser.read(PAGE, allow_private=True)[2])

    def test_refusals_are_put_plainly(self):
        self.assertIn("local network", browser._plainly("Reached error page: about:neterror?e=proxyConnectFailure&u=x"))
        self.assertEqual(browser._plainly("Reached error page: about:neterror?e=dnsNotFound&u=x"), "the page could not be opened (dnsNotFound)")


class UsingAPageTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.profile = Path(tmp.name) / "profile"
        self.browser = stand_in(self, tmp.name)

    def did(self):
        return (self.profile / "did").read_text().splitlines()

    def test_an_open_page_lists_what_can_be_used_and_stays_open(self):
        view = self.browser.open("me", "https://example.org/shop")
        self.assertEqual((view.url, view.title, view.host), ("https://example.org/shop", "Shop", "example.org"))
        self.assertIn('[3 button "Search"]', view.text)
        self.assertEqual(view.item("1")["href"], "https://example.org/shop/2")
        self.assertEqual(self.browser.read("https://example.org/other")[0], "https://example.org/other")  # in its own tab
        self.assertEqual(self.browser.look("me").url, "https://example.org/shop")

    def test_clicking_and_typing(self):
        self.browser.open("me", "https://example.org/shop")
        self.assertEqual(self.browser.click("me", 1).url, "https://example.org/shop/2")
        self.assertEqual(self.browser.fill("me", 2, "red\nshoes", enter=True).url, "https://example.org/found")
        self.browser.fill("me", 6, "Blue")
        self.assertEqual(self.did(), ["clicked 1", "typed into 2: 'red shoes\\ue006'", "chose Blue"])
        with self.assertRaisesRegex(browser.BrowserError, "no choice like"):
            self.browser.fill("me", 6, "Green")

    def test_a_password_is_never_typed(self):
        self.browser.open("me", "https://example.org/shop")
        with self.assertRaisesRegex(browser.BrowserError, "only the user"):
            self.browser.fill("me", 5, "hunter2")
        with self.assertRaisesRegex(browser.BrowserError, "use browser_click"):
            self.browser.fill("me", 3, "x")
        self.assertFalse((self.profile / "did").exists())

    def test_a_number_from_an_older_look_or_one_that_is_not_there(self):
        view = self.browser.open("me", "https://example.org/shop")
        with self.assertRaisesRegex(browser.BrowserError, "no number 9"):
            self.browser.click("me", 9)
        self.browser.look("me")
        self.browser.views["me"] = view  # as if the page had changed under the numbers
        with self.assertRaisesRegex(browser.BrowserError, "no longer on the page"):
            self.browser.click("me", 1)

    def test_a_click_that_leads_inside_is_not_followed(self):
        self.browser.open("me", "https://example.org/trap")
        with self.assertRaisesRegex(FetchError, "only public"):
            self.browser.click("me", 3)
        with self.assertRaises(browser.BrowserError):
            self.browser.page("me")

    def test_each_user_of_a_page_has_their_own_and_none_without_opening(self):
        self.browser.open("me", "https://example.org/shop")
        self.browser.open("you", "https://example.org/other-shop")
        self.browser.look("me")  # the stand-in numbers things anew at every look, whoever looks
        self.assertEqual(self.browser.click("me", 1).url, "https://example.org/shop/2")
        self.assertEqual(self.browser.look("you").url, "https://example.org/other-shop")
        with self.assertRaisesRegex(browser.BrowserError, "no page is open"):
            self.browser.look("someone else")

    def test_when_the_browser_has_closed_the_page_has_to_be_opened_again(self):
        self.browser.open("me", "https://example.org/shop")
        self.browser.close()
        with self.assertRaisesRegex(browser.BrowserError, "no page is open"):
            self.browser.click("me", 1)

    def test_what_makes_a_click_worth_a_warning(self):
        self.assertIn("Place order", browser.consequential({"label": "Place order"}))
        self.assertIn("password", browser.consequential({"label": "Continue", "risky": True}))
        for label in ("Search", "Next page", "Sort order", "Reject all", "Postcode lookup"):
            self.assertIsNone(browser.consequential({"label": label}), label)


class SettingsTest(unittest.TestCase):
    def test_a_snap_firefox_gets_a_profile_it_is_able_to_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "firefox"
            script.write_text("#!/bin/sh\nexec /snap/bin/firefox \"$@\"\n")
            self.assertEqual(browser.profile_dir(str(script), {}), Path.home() / "snap/firefox/common/private_ai-browser")
            script.write_text("#!/bin/sh\nexec /opt/firefox/firefox \"$@\"\n")
            with mock.patch.dict(os.environ, {"PRIVATE_AI_DATA": tmp}):
                self.assertEqual(browser.profile_dir(str(script), {}), Path(tmp) / "browser")
            self.assertEqual(browser.profile_dir(str(script), {"profile": "~/elsewhere"}), Path.home() / "elsewhere")

    def test_no_browser_when_switched_off_or_not_installed(self):
        with mock.patch.object(browser, "_shared", None):
            self.assertIsNone(browser.shared())  # tests run with PRIVATE_AI_BROWSER=off
            with self.assertRaisesRegex(browser.BrowserError, "no browser"):
                browser.read_page("https://example.org/")
            with mock.patch.dict(os.environ, {"PRIVATE_AI_BROWSER": ""}), mock.patch.object(browser.shutil, "which", return_value=None):
                self.assertIsNone(browser.shared())
                with mock.patch.object(browser, "settings", return_value={"enabled": False}):
                    self.assertIsNone(browser.shared())


class UsedByTheAssistantTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = Store(Path(tmp.name))
        self.ui = FakeUI()
        self.box = tools.Toolbox(self.store, self.ui, Provider(name="local", base_url="http://127.0.0.1:1/v1"))
        self.box.known_urls.add("https://example.org/app")
        self.seen = []

    def with_browser(self, result):
        def read(url):
            self.seen.append(url)
            if isinstance(result, Exception):
                raise result
            return result

        return mock.patch.multiple(tools.browser_module, shared=lambda: object(), read_page=read)

    def test_a_page_that_cannot_be_read_plainly_is_read_with_the_browser(self):
        refused = mock.patch.object(tools, "read_page", side_effect=FetchError("the site answered 403 Forbidden"))
        with refused, self.with_browser(("https://example.org/app", "App", "built by scripts " * 30)):
            result = self.box.run("fetch_page", '{"url": "https://example.org/app"}')
        self.assertIn("built by scripts", result)
        self.assertIn(tools.UNTRUSTED, result)
        self.assertEqual(self.ui.notices, ["read https://example.org/app with the browser"])
        self.assertEqual(self.store.load_page("https://example.org/app")["title"], "App")

    def test_the_model_can_ask_for_the_browser_when_the_plain_copy_was_not_the_page(self):
        plain = mock.patch.object(tools, "read_page", side_effect=AssertionError("not to be used"))
        with plain, self.with_browser(("https://example.org/app", "App", "the real page " * 30)):
            result = self.box.run("fetch_page", '{"url": "https://example.org/app", "browser": "yes"}')
        self.assertIn("the real page", result)

    def test_an_address_the_model_made_up_is_still_asked_about_first(self):
        self.ui.answers = ["n"]
        with self.with_browser(("x", "x", "x" * 500)):
            result = self.box.run("fetch_page", '{"url": "https://made-up.example/x", "browser": "yes"}')
        self.assertIn("did not allow", result)
        self.assertEqual(self.seen, [])

    def test_when_the_browser_fails_too_both_reasons_are_given(self):
        refused = mock.patch.object(tools, "read_page", side_effect=FetchError("the site answered 403 Forbidden"))
        no_reader = mock.patch.object(tools.search, "load_backends", return_value={})
        with refused, no_reader, self.with_browser(browser.BrowserError("it may need a login")):
            result = self.box.run("fetch_page", '{"url": "https://example.org/app"}')
        self.assertIn("403 Forbidden; the browser could not read it either: it may need a login", result)

    def test_without_a_browser_nothing_changes(self):
        refused = mock.patch.object(tools, "read_page", side_effect=FetchError("the site answered 403 Forbidden"))
        no_reader = mock.patch.object(tools.search, "load_backends", return_value={})
        with refused, no_reader:
            result = self.box.run("fetch_page", '{"url": "https://example.org/app"}')
        self.assertIn("403 Forbidden. Rely on the excerpt", result)


class UsingAPageWithTheAssistantTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.profile = Path(tmp.name) / "profile"
        self.browser = stand_in(self, tmp.name)
        patch = mock.patch.object(tools.browser_module, "shared", lambda: self.browser)
        patch.start()
        self.addCleanup(patch.stop)
        config = mock.patch.dict(os.environ, {"PRIVATE_AI_CONFIG": str(Path(tmp.name) / "config" / "config.toml")})
        config.start()
        self.addCleanup(config.stop)
        self.store = Store(Path(tmp.name) / "data")
        self.ui = FakeUI()
        self.box = self.toolbox()
        self.box.known_urls.add("https://example.org/shop")

    def toolbox(self, trust="ask"):
        return tools.Toolbox(self.store, self.ui, Provider(name="local", base_url="http://127.0.0.1:1/v1"), trust=trust)

    def did(self):
        path = self.profile / "did"
        return path.read_text().splitlines() if path.exists() else []

    def open(self):
        return self.box.run("browser_open", '{"url": "https://example.org/shop"}')

    def test_the_tools_are_there_only_with_a_browser(self):
        names = {t["function"]["name"] for t in self.box.tools}
        self.assertLessEqual({"browser_open", "browser_click", "browser_type", "browser_look"}, names)
        with mock.patch.object(tools.browser_module, "shared", lambda: None):
            self.assertNotIn("browser_open", {t["function"]["name"] for t in self.toolbox().tools})

    def test_opening_shows_the_page_marked_as_untrusted(self):
        result = self.open()
        self.assertIn("Address: https://example.org/shop", result)
        self.assertIn(tools.UNTRUSTED, result)
        self.assertIn('[4 button "Place order"]', result)
        self.assertEqual(self.ui.questions, [])
        self.assertEqual(self.ui.notices, ["opened https://example.org/shop in the browser"])

    def test_an_address_the_model_made_up_is_asked_about_before_it_is_opened(self):
        self.ui.answers = ["n"]
        self.assertIn("did not allow", self.box.run("browser_open", '{"url": "https://example.org/?q=private+thing"}'))
        self.assertFalse((self.profile / "opened").exists())

    def test_a_click_is_asked_about_and_says_where_it_leads(self):
        self.open()
        self.ui.answers = ["n", "y"]
        self.assertIn("did not allow", self.box.run("browser_click", '{"number": "1"}'))
        self.assertEqual(self.did(), [])
        question = self.ui.questions[0]
        self.assertIn('click this on example.org?\n  link "Next page"\nIt leads to https://example.org/shop/2', question)
        result = self.box.run("browser_click", '{"number": "1"}')
        self.assertIn("Address: https://example.org/shop/2", result)
        self.assertEqual(self.did(), ["clicked 1"])
        self.assertEqual(self.ui.notices[-1], 'clicked link "Next page" on example.org')

    def test_a_site_can_be_approved_for_the_session_but_not_what_looks_like_an_order(self):
        self.open()
        self.ui.answers = ["s", "n"]
        self.box.run("browser_click", '{"number": "3"}')
        self.box.run("browser_click", '{"number": "3"}')  # not asked again
        self.assertEqual(len(self.ui.questions), 1)
        self.assertIn("did not allow", self.box.run("browser_click", '{"number": "4"}'))
        self.assertIn("WARNING", self.ui.questions[1])
        self.assertIn('It says "Place order"', self.ui.questions[1])
        self.assertEqual(self.did(), ["clicked 3", "clicked 3"])

    def test_an_approval_for_one_site_is_not_one_for_another(self):
        self.box.policy.grant("click", "example.org", "session")
        self.assertEqual(self.box.policy.decide("click", "www.example.org")[0], "allow")
        self.assertEqual(self.box.policy.decide("click", "example.org.evil.test")[0], "ask")
        self.assertEqual(self.box.policy.decide("type", "example.org")[0], "ask")

    def test_typing_shows_the_words_first(self):
        self.open()
        self.ui.answers = ["y"]
        result = self.box.run("browser_type", '{"number": "2", "text": "red shoes", "enter": "yes"}')
        self.assertIn("Address: https://example.org/found", result)
        question = self.ui.questions[0]
        self.assertIn('type this into the field "Search" on example.org?\n  red shoes\nEnter will be pressed', question)
        self.assertEqual(self.did(), ["typed into 2: 'red shoes\\ue006'"])
        self.assertEqual(self.ui.notices[-1], 'typed into field "Search" on example.org: red shoes')

    def test_typing_that_is_refused_types_nothing(self):
        self.open()
        self.ui.answers = ["n"]
        self.assertIn("did not allow", self.box.run("browser_type", '{"number": "2", "text": "my address"}'))
        self.assertEqual(self.did(), [])

    def test_a_password_is_refused_without_a_question(self):
        self.open()
        result = self.box.run("browser_type", '{"number": "5", "text": "hunter2"}')
        self.assertIn("only the user may fill in", result)
        self.assertEqual((self.ui.questions, self.did()), ([], []))

    def test_rules_in_the_file_cover_sites(self):
        self.open()
        self.box.policy.path.parent.mkdir(parents=True)
        self.box.policy.path.write_text(
            '[[rule]]\ndecision = "allow"\naction = "type"\nmatch = "example.org"\n'
            '[[rule]]\ndecision = "deny"\naction = "click"\nmatch = "*"\n'
        )
        self.box.run("browser_type", '{"number": "6", "text": "Red"}')
        self.assertIn("rules forbid", self.box.run("browser_click", '{"number": "1"}'))
        self.assertEqual((self.ui.questions, self.did()), ([], ["chose Red"]))

    def test_full_trust_asks_nothing(self):
        self.box = self.toolbox("all")
        self.box.known_urls.add("https://example.org/shop")
        self.open()
        self.box.run("browser_click", '{"number": "4"}')
        self.assertEqual((self.ui.questions, self.did()), ([], ["clicked 4"]))

    def test_a_long_page_comes_in_parts(self):
        self.box.known_urls.add("https://example.org/long")
        first = self.box.run("browser_open", '{"url": "https://example.org/long"}')
        self.assertIn("[part 1 of ", first)
        self.assertLess(len(first), tools.MAX_PAGE_CHARS + 600)
        second = self.box.run("browser_look", '{"part": "2"}')
        self.assertIn("[part 2 of ", second)
        self.assertNotIn("[1 link", second)
        self.assertIn("part(s)", self.box.run("browser_look", '{"part": "999"}'))

    def test_nothing_is_open_until_something_is_opened(self):
        self.assertIn("no page is open", self.box.run("browser_click", '{"number": "1"}'))
        self.assertIn("no page is open", self.box.run("browser_look", "{}"))


if __name__ == "__main__":
    unittest.main()
