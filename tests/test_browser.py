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


if __name__ == "__main__":
    unittest.main()
