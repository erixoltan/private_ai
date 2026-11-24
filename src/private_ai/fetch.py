"""Fetch a web page and reduce it to readable text, using only the standard library.

Privacy and safety properties this module is responsible for:

* Only public internet addresses are contacted. A page address that resolves
  to this machine or the local network is refused, and the connection is made
  to the address that was checked, so a name cannot change between the check
  and the request. Every redirect is checked the same way.
* The request carries nothing about the user: no cookies, no referrer, and a
  User-Agent that only names this program.
* Nothing is executed. Pages that need JavaScript come back nearly empty and
  are reported as unreadable rather than rendered.
"""

from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
import zlib
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from private_ai import __version__

MAX_BYTES = 3_000_000
MAX_REDIRECTS = 5
READABLE_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "text/markdown", "application/json")
FEED_TYPES = ("application/rss+xml", "application/atom+xml", "application/xml", "text/xml")
THIN_PAGE_CHARS = 200  # less text than this usually means the page is built by JavaScript


class FetchError(Exception):
    """A page could not be fetched or read. The message is safe to show."""


def _public_address(host: str, port: int, allow_private: bool) -> str:
    try:
        found = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        raise FetchError(f"cannot find the site {host}") from None
    addresses = [info[4][0] for info in found]
    if not allow_private:
        for address in addresses:
            ip = ipaddress.ip_address(address.split("%")[0])
            ip = getattr(ip, "ipv4_mapped", None) or ip
            if not ip.is_global:
                raise FetchError(f"{host} is on this machine or the local network; only public sites can be fetched")
    return addresses[0]


class _PinnedHTTP(http.client.HTTPConnection):
    """Connects to an address that has already been checked, whatever the name now resolves to."""

    address = ""

    def connect(self):
        self.sock = socket.create_connection((self.address, self.port), self.timeout)


class _PinnedHTTPS(http.client.HTTPSConnection):
    address = ""
    tls = ssl.create_default_context()

    def connect(self):
        sock = socket.create_connection((self.address, self.port), self.timeout)
        self.sock = self.tls.wrap_socket(sock, server_hostname=self.host)


def _get(url: str, timeout: float, allow_private: bool, accept: str):
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("only http and https addresses can be fetched")
    if parts.username or parts.password:
        raise FetchError("addresses containing a user name or password are not fetched")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        raise FetchError("that address has an invalid port") from None
    connection = (_PinnedHTTPS if parts.scheme == "https" else _PinnedHTTP)(parts.hostname, port, timeout=timeout)
    connection.address = _public_address(parts.hostname, port, allow_private)
    target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    connection.request("GET", target, headers={"User-Agent": f"private_ai/{__version__}", "Accept": accept})
    return connection, connection.getresponse()


def fetch(url: str, timeout: float = 20, allow_private: bool = False, feeds: bool = False) -> tuple[str, str, bytes]:
    """GET a page. Returns (final address, content type, body)."""
    wanted = FEED_TYPES + READABLE_TYPES if feeds else READABLE_TYPES
    accept = ", ".join(wanted) + ", */*;q=0.1"
    for _ in range(MAX_REDIRECTS + 1):
        connection = None
        try:
            connection, response = _get(url, timeout, allow_private, accept)
            if response.status in (301, 302, 303, 307, 308) and response.getheader("Location"):
                url = urljoin(url, response.getheader("Location"))
                continue
            if response.status != 200:
                raise FetchError(f"the site answered {response.status} {response.reason}")
            content_type = (response.getheader("Content-Type") or "text/html").lower()
            kind = content_type.split(";")[0].strip()
            if kind == "application/pdf":
                raise FetchError("that is a PDF, which this program cannot read yet")
            if kind not in wanted:
                raise FetchError(f"that is not a readable page (it is {kind})")
            body = response.read(MAX_BYTES + 1)
            if (response.getheader("Content-Encoding") or "").lower() in ("gzip", "deflate"):
                body = zlib.decompressobj(47).decompress(body, MAX_BYTES + 1)
            return url, content_type, body[:MAX_BYTES]
        except FetchError:
            raise
        except TimeoutError:
            raise FetchError(f"the site did not answer within {timeout:g} seconds") from None
        except ssl.SSLError as e:
            raise FetchError(f"secure connection failed: {e.reason or e}") from None
        except (OSError, http.client.HTTPException, zlib.error) as e:
            raise FetchError(f"could not fetch the page: {e}") from None
        finally:
            if connection:
                connection.close()
    raise FetchError("too many redirects")


# ---- turning HTML into text ----

_SKIP = {"script", "style", "noscript", "template", "svg", "iframe", "nav", "footer", "aside", "form", "button", "select"}
_ONLY_CODE = {"script", "style", "noscript", "template", "svg"}
_BLOCK = {
    "p", "div", "section", "article", "main", "header", "br", "tr", "table", "ul", "ol", "blockquote", "pre",
    "figure", "figcaption", "dl", "dt", "dd", "hr",
}  # fmt: skip
_HEADINGS = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ", "h5": "#### ", "h6": "#### "}


class _Reader(HTMLParser):
    def __init__(self, skip: set[str]):
        super().__init__(convert_charrefs=True)
        self.skip = skip
        self.skipping = 0
        self.in_title = False
        self.titled = False  # only the first <title> is the page's; later ones belong to images
        self.in_main = 0
        self.title: list[str] = []
        self.everything: list[str] = []
        self.main: list[str] = []

    def _add(self, text: str) -> None:
        self.everything.append(text)
        if self.in_main:
            self.main.append(text)

    def handle_starttag(self, tag, attrs):
        if tag in self.skip:
            self.skipping += 1
        elif tag == "title":
            self.in_title = not self.skipping and not self.titled
        elif not self.skipping:
            if tag in ("main", "article"):
                self.in_main += 1
            if tag in _HEADINGS:
                self._add("\n\n" + _HEADINGS[tag])
            elif tag == "li":
                self._add("\n- ")
            elif tag in _BLOCK:
                self._add("\n")

    def handle_endtag(self, tag):
        if tag in self.skip:
            self.skipping = max(0, self.skipping - 1)
        elif tag == "title":
            self.titled = self.titled or self.in_title
            self.in_title = False
        elif not self.skipping:
            if tag in _HEADINGS or tag in _BLOCK or tag == "li":
                self._add("\n")
            if tag in ("main", "article"):
                self.in_main = max(0, self.in_main - 1)

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        elif not self.skipping:
            self._add(data)


def _tidy(pieces: list[str]) -> str:
    lines = [" ".join(line.split()) for line in "".join(pieces).split("\n")]
    text = "\n".join(line for line in lines if line.strip("-# ") or not line)  # no empty bullets or headings
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _decode(body: bytes, content_type: str) -> str:
    declared = re.search(r"charset=([\w-]+)", content_type) or re.search(
        rb"<meta[^>]+charset=[\"']?([\w-]+)", body[:4096], re.I
    )
    name = declared.group(1) if declared else "utf-8"
    if isinstance(name, bytes):
        name = name.decode("ascii", "replace")
    try:
        return body.decode(name, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def strip_tags(html: str) -> str:
    """A fragment of HTML (a search snippet, a feed summary) as one line of text."""
    reader = _Reader(_ONLY_CODE)
    reader.feed(html)
    return " ".join("".join(reader.everything).split())


def extract(body: bytes, content_type: str) -> tuple[str, str]:
    """(title, readable text) of a fetched page."""
    source = _decode(body, content_type)
    if "html" not in content_type and not re.match(r"\s*<(!doctype|html)", source[:200], re.I):
        return "", source.strip()

    def read(skip: set[str]) -> tuple[str, str]:
        reader = _Reader(skip)
        reader.feed(source)
        reader.close()
        whole, main = _tidy(reader.everything), _tidy(reader.main)
        # Prefer the part the page itself marks as its content, when it is substantial.
        return " ".join("".join(reader.title).split()), main if len(main) >= 600 else whole

    title, text = read(_SKIP)
    if len(text) < THIN_PAGE_CHARS:
        # A badly nested page can hide everything behind one unclosed tag; read it less strictly.
        title, text = read(_ONLY_CODE)
    return title, text


def read_page(url: str, timeout: float = 20, allow_private: bool = False) -> tuple[str, str, str]:
    """Fetch a page and return (final address, title, readable text)."""
    final_url, content_type, body = fetch(url, timeout, allow_private)
    title, text = extract(body, content_type)
    if len(text) < THIN_PAGE_CHARS:
        raise FetchError("the page has almost no readable text; it probably needs JavaScript or a login")
    return final_url, title, text
