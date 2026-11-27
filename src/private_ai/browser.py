"""A real browser, for the pages that plain fetching cannot read.

fetch.py reads a page without running anything, so a page built by JavaScript
comes back empty and a site that refuses programs refuses it. This module
opens such a page in Firefox, lets it finish building itself, and hands back
the same readable text.

How it works: Firefox has a remote-control protocol of its own, Marionette,
which is a plain socket carrying JSON. That is all this needs, so there is no
driver program and nothing to install besides Firefox.

Privacy and safety properties this module is responsible for:

* The browser has a profile of its own, apart from the one the user browses
  with. It starts signed in to nothing. The user can sign in to sites in it
  (`private-ai browser` opens it as an ordinary window), and those sign-ins
  are then used when the assistant reads those sites. The user's everyday
  history, passwords and sessions are in another profile, out of reach.
* Unlike fetch.py, pages run here: their scripts run, and they load what they
  like from other sites, as in any browser. Cookies in this profile are sent.
* Only public internet addresses are opened. The address is checked before it
  is opened and again where the page ended up, and the profile is set up so
  that Firefox itself refuses to connect to this machine or the local network,
  which covers redirects and whatever a page tries to load.
* The assistant can only look. Nothing here clicks, types or submits.
* Text from a page is as untrusted as any other text from the web.

Settings, in the config file:

    [browser]
    enabled = false                      # default: used whenever Firefox is installed
                                         # (PRIVATE_AI_BROWSER=off turns it off for one run)
    firefox = "/path/to/firefox"         # default: firefox on the PATH
    profile = "/path/to/profile"         # default: see profile_dir
    headless = false                     # default true: no window while the assistant reads
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

from private_ai.fetch import THIN_PAGE_CHARS, FetchError, _public_address, extract
from private_ai.providers import config_path
from private_ai.store import data_dir

START_SECONDS = 40  # a first start creates the profile, which is slow
LOAD_SECONDS = 30
SETTLE_SECONDS = 8  # how long a page is given to finish building itself after it has loaded
IDLE_SECONDS = 120  # the browser is closed when it has not been used for this long
MAX_HTML_CHARS = 3_000_000

# Sent to nothing on this machine or the local network, whatever a page asks for. Firefox
# runs this for every request; a refused one is sent to a proxy that does not exist.
_PUBLIC_ONLY = """
function FindProxyForURL(url, host) {
  var no = "PROXY 127.0.0.1:1";
  host = host.toLowerCase().replace(/^\\[|\\]$/g, "");
  if (host == "localhost" || /\\.(localhost|local|internal|lan|home|intranet)$/.test(host) || host.indexOf(".") < 0 && host.indexOf(":") < 0) return no;
  if (host.indexOf(":") >= 0) return /^(::1?$|f[cde]|::ffff:)/.test(host) ? no : "DIRECT";
  var ip = dnsResolve(host);
  if (!ip) return "DIRECT";
  var nets = [["0.0.0.0", "255.0.0.0"], ["10.0.0.0", "255.0.0.0"], ["100.64.0.0", "255.192.0.0"],
    ["127.0.0.0", "255.0.0.0"], ["169.254.0.0", "255.255.0.0"], ["172.16.0.0", "255.240.0.0"],
    ["192.168.0.0", "255.255.0.0"], ["224.0.0.0", "224.0.0.0"]];
  for (var i = 0; i < nets.length; i++) if (isInNet(ip, nets[i][0], nets[i][1])) return no;
  return "DIRECT";
}
"""

# A quiet browser that keeps to itself: no first-run pages, no reports home, no saved passwords.
_PREFS = {
    "browser.shell.checkDefaultBrowser": False,
    "browser.aboutwelcome.enabled": False,
    "browser.startup.homepage": "about:blank",
    "browser.startup.homepage_override.mstone": "ignore",
    "startup.homepage_welcome_url": "",
    "startup.homepage_welcome_url.additional": "",
    "browser.newtabpage.enabled": False,
    "browser.sessionstore.resume_from_crash": False,
    "datareporting.policy.dataSubmissionEnabled": False,
    "datareporting.healthreport.uploadEnabled": False,
    "toolkit.telemetry.enabled": False,
    "toolkit.telemetry.unified": False,
    "app.normandy.enabled": False,
    "app.shield.optoutstudies.enabled": False,
    "browser.discovery.enabled": False,
    "extensions.pocket.enabled": False,
    "signon.rememberSignons": False,
    "media.autoplay.default": 5,
    "permissions.default.geo": 2,
    "permissions.default.camera": 2,
    "permissions.default.microphone": 2,
    "permissions.default.desktop-notification": 2,
    "network.proxy.type": 2,
    "network.proxy.no_proxies_on": "",
    "network.proxy.allow_hijacking_localhost": True,
    "network.proxy.failover_direct": False,
}

# What the page shows, when the HTML alone does not say (text drawn into the page by scripts in odd ways).
_SHOWN = "return [document.title, document.body ? document.body.innerText : '']"
_SIZE = "return document.body ? document.body.innerText.length : 0"
_HTML = "return document.documentElement ? document.documentElement.outerHTML.slice(0, arguments[0]) : ''"


_CHALLENGE = re.compile(r"captcha|cf-chl|challenge-platform|are you a robot|verify you are human", re.I)


class BrowserError(FetchError):
    """The browser could not read a page. The message is safe to show."""


def settings(path: Path | None = None) -> dict:
    path = path or config_path()
    try:
        with path.open("rb") as f:
            found = tomllib.load(f).get("browser", {})
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def firefox_program(config: dict | None = None) -> str | None:
    config = settings() if config is None else config
    return shutil.which(os.path.expanduser(str(config.get("firefox") or "firefox")))


def _is_snap(program: str) -> bool:
    """Whether this Firefox is the snap package, which can only use folders inside its own."""
    if "/snap/" in os.path.realpath(program):
        return True
    try:
        if os.path.getsize(program) < 10_000:  # Ubuntu's /usr/bin/firefox is a small script that starts the snap
            return b"/snap/bin/firefox" in Path(program).read_bytes()
    except OSError:
        pass
    return False


def profile_dir(program: str | None = None, config: dict | None = None) -> Path:
    """Where the assistant's browser profile is kept. With the program's other
    data, unless Firefox is a snap package: that is confined to its own folder
    and cannot open a profile anywhere else."""
    config = settings() if config is None else config
    if config.get("profile"):
        return Path(os.path.expanduser(str(config["profile"])))
    if program and _is_snap(program):
        return Path.home() / "snap" / "firefox" / "common" / "private_ai-browser"
    return data_dir() / "browser"


def _write_prefs(profile: Path, port: int | None) -> None:
    profile.mkdir(parents=True, exist_ok=True, mode=0o700)
    downloads = profile / "downloads"  # nothing a page starts downloading lands among the user's files
    prefs = dict(_PREFS) | {
        "network.proxy.autoconfig_url": "data:application/x-ns-proxy-autoconfig;base64,"
        + base64.b64encode(_PUBLIC_ONLY.encode()).decode(),
        "browser.download.folderList": 2,
        "browser.download.dir": str(downloads),
    }
    if port:
        prefs["marionette.port"] = port
    lines = [f"user_pref({json.dumps(name)}, {json.dumps(value)});" for name, value in prefs.items()]
    (profile / "user.js").write_text("// Written by private_ai each time it starts this browser.\n" + "\n".join(lines) + "\n")


def _check_public(url: str, allow_private: bool) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BrowserError("only http and https addresses can be opened")
    _public_address(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80), allow_private)


def _plainly(message: str) -> str:
    """What the browser said went wrong, in words fit to show."""
    if "proxyConnectFailure" in message:  # how _PUBLIC_ONLY's refusal comes back
        return "the address is on this machine or the local network; only public sites can be opened"
    if "about:neterror?e=" in message:
        return f"the page could not be opened ({message.split('about:neterror?e=')[1].split('&')[0]})"
    return message[:300] or "the browser reported an error"


class Browser:
    """One Firefox, started when first needed and closed when idle. Safe to use from several threads; they take turns."""

    def __init__(
        self,
        program: str,
        profile: Path,
        headless: bool = True,
        idle_seconds: float = IDLE_SECONDS,
        settle_seconds: float = SETTLE_SECONDS,
    ):
        self.program = program
        self.profile = profile
        self.headless = headless
        self.idle_seconds = idle_seconds
        self.settle_seconds = settle_seconds
        self.lock = threading.RLock()
        self.process: subprocess.Popen | None = None
        self.sock: socket.socket | None = None
        self.buffer = b""
        self.sent = 0
        self.timer: threading.Timer | None = None

    # ---- the wire: "length:json", requests [0, id, name, params], replies [1, id, error, result] ----

    def _receive(self) -> list:
        while b":" not in self.buffer:
            self._more()
        size, _, rest = self.buffer.partition(b":")
        self.buffer = rest
        while len(self.buffer) < int(size):
            self._more()
        message, self.buffer = self.buffer[: int(size)], self.buffer[int(size) :]
        return json.loads(message)

    def _more(self) -> None:
        piece = self.sock.recv(65536)
        if not piece:
            raise ConnectionError("the browser closed the connection")
        self.buffer += piece

    def _command(self, name: str, **params):
        self.sent += 1
        data = json.dumps([0, self.sent, name, params]).encode()
        self.sock.sendall(str(len(data)).encode() + b":" + data)
        while True:
            reply = self._receive()
            if reply[0] == 1 and reply[1] == self.sent:
                break
        if reply[2]:
            raise BrowserError(_plainly(str(reply[2].get("message") or reply[2].get("error") or "")))
        result = reply[3]
        return result.get("value") if isinstance(result, dict) and set(result) == {"value"} else result

    def _script(self, script: str, *args):
        return self._command("WebDriver:ExecuteScript", script=script, args=list(args))

    # ---- starting and stopping ----

    def _start(self) -> None:
        if self.process and self.process.poll() is None and self.sock:
            return
        self.close()
        with socket.socket() as probe:  # a port nothing else is using
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        try:
            _write_prefs(self.profile, port)
            command = [self.program, "--marionette", "--no-remote", "--profile", str(self.profile)]
            self.process = subprocess.Popen(
                command + (["--headless"] if self.headless else []),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # its own group, so every process it starts can be ended with it
            )
        except OSError as e:
            raise BrowserError(f"the browser could not be started: {e}") from None
        deadline = time.monotonic() + START_SECONDS
        while self.sock is None:
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), 1)
            except OSError:
                if self.process.poll() is not None:
                    self.close()
                    raise BrowserError(
                        "the browser would not start. If its window is open (private-ai browser), close it first"
                    ) from None
                if time.monotonic() > deadline:
                    self.close()
                    raise BrowserError("the browser did not start in time") from None
                time.sleep(0.1)
        try:
            self.sock.settimeout(LOAD_SECONDS + 15)
            self.buffer, self.sent = b"", 0
            self._receive()  # its greeting
            self._command("WebDriver:NewSession", capabilities={"alwaysMatch": {"unhandledPromptBehavior": "dismiss"}})
            self._command("WebDriver:SetTimeouts", pageLoad=LOAD_SECONDS * 1000, script=10_000)
        except (OSError, ValueError, BrowserError) as e:
            self.close()
            raise BrowserError(f"the browser started but could not be controlled: {e}") from None

    def close(self) -> None:
        with self.lock:
            if self.timer:
                self.timer.cancel()
                self.timer = None
            if self.sock:
                try:
                    self.sock.settimeout(5)
                    self._command("Marionette:Quit", flags=["eForceQuit"])  # lets it save cookies properly
                except (OSError, ValueError, BrowserError):
                    pass
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None
            if self.process:
                try:
                    self.process.wait(5 if self.process.poll() is None else 0)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)  # whatever is left of it
                except OSError:
                    pass
                self.process.wait()
                self.process = None

    def _close_when_idle(self) -> None:
        if self.timer:
            self.timer.cancel()
        if self.idle_seconds:
            self.timer = threading.Timer(self.idle_seconds, self.close)
            self.timer.daemon = True
            self.timer.start()

    # ---- reading ----

    def read(self, url: str, allow_private: bool = False) -> tuple[str, str, str]:
        """Open a page, let it finish, and return (final address, title, readable text)."""
        _check_public(url, allow_private)
        with self.lock:
            self._start()
            try:
                return self._read(url, allow_private)
            except (OSError, ValueError) as e:  # the browser died or stopped answering
                self.close()
                raise BrowserError(f"the browser stopped responding: {e}") from None
            finally:
                if self.sock:
                    try:
                        self._command("WebDriver:Navigate", url="about:blank")  # the page stops running once it is read
                    except (OSError, ValueError, BrowserError):
                        self.close()
                self._close_when_idle()

    def _read(self, url: str, allow_private: bool) -> tuple[str, str, str]:
        problem = ""
        try:
            self._command("WebDriver:Navigate", url=url)
        except BrowserError as e:
            problem = str(e)  # too slow, perhaps: what did arrive may still be worth reading
        final_url = str(self._command("WebDriver:GetCurrentURL") or "")
        if not final_url.startswith(("http://", "https://")):
            raise BrowserError(problem or "the page did not open")
        _check_public(final_url, allow_private)
        # A page built by scripts keeps growing after it has "loaded". Wait until it holds still.
        deadline, size, same = time.monotonic() + self.settle_seconds, -1, 0
        while time.monotonic() < deadline and same < 2:
            time.sleep(0.4)
            now = self._script(_SIZE)
            same, size = (same + 1 if now == size else 0), now
        html = self._script(_HTML, MAX_HTML_CHARS) or ""
        title, text = extract(html.encode(), "text/html; charset=utf-8")
        shown_title, shown = self._script(_SHOWN) or ["", ""]
        shown = "\n".join(line.strip() for line in str(shown).splitlines() if line.strip())
        if len(text) < max(THIN_PAGE_CHARS, len(shown) // 3):
            text = shown  # the HTML reader missed most of what the page shows
        if len(text) < THIN_PAGE_CHARS:
            if _CHALLENGE.search(html):
                # Not ours to answer. The user can, in the window, and the site may then remember this browser.
                raise BrowserError(
                    "the site wants proof that a person is there (a captcha). The user can open it with "
                    f"`private-ai browser {url}` and answer it themselves, which may let later reads through"
                )
            raise BrowserError(problem or "the page has almost no readable text, even in a browser; it may need a login")
        return final_url, title or str(shown_title or ""), text


_shared: Browser | None = None
_shared_lock = threading.Lock()


def shared() -> Browser | None:
    """The browser the assistant reads with, or None if there is none to use."""
    global _shared
    with _shared_lock:
        if _shared is None:
            config = settings()
            off = config.get("enabled", True) is False or os.environ.get("PRIVATE_AI_BROWSER", "").lower() in ("off", "0", "no")
            program = None if off else firefox_program(config)
            if not program:
                return None
            _shared = Browser(program, profile_dir(program, config), config.get("headless", True) is not False)
            atexit.register(_shared.close)
        return _shared


def read_page(url: str) -> tuple[str, str, str]:
    """Read a page with the browser. Raises BrowserError if there is no browser or it cannot."""
    browser = shared()
    if browser is None:
        raise BrowserError("no browser is set up")
    return browser.read(url)


def open_window(url: str | None = None) -> int:
    """Open the assistant's browser profile as an ordinary window, for the user
    to sign in to sites or look at what is there. Returns when it is closed."""
    config = settings()
    program = firefox_program(config)
    if not program:
        raise BrowserError("Firefox was not found. Install it, or set firefox = \"...\" under [browser] in the config file")
    profile = profile_dir(program, config)
    _write_prefs(profile, None)
    return subprocess.call([program, "--no-remote", "--profile", str(profile), *([url] if url else [])])
