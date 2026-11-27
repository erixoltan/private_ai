"""Stands in for Firefox in tests: speaks enough of its remote-control protocol
to be started, asked for pages and told to quit. What a page holds depends on
words in its address: "thin" has almost no text, "slow" fails to finish
loading, "drawn" has its text only on screen and not in its HTML, and
"wanders" ends up at an address on this machine, and "guarded" shows a
captcha in place of the page.

A page opened to be used always has the same six things on it (see THINGS).
Clicking the link goes where it leads; on a page with "trap" in its address,
clicking the first button goes to an address on this machine. What was
clicked and typed is written to the file "did" in the profile."""

import json
import re
import socket
import sys
from pathlib import Path

profile = Path(sys.argv[sys.argv.index("--profile") + 1])
port = int(re.search(r'"marionette.port", (\d+)', (profile / "user.js").read_text())[1])
(profile / "started").write_text(" ".join(sys.argv[1:]))
server = socket.create_server(("127.0.0.1", port))
connection, _ = server.accept()


def send(message):
    data = json.dumps(message).encode()
    connection.sendall(str(len(data)).encode() + b":" + data)


ELEMENT = "element-6066-11e4-a52e-4f735466cecf"
THINGS = [
    {"kind": "link", "label": "Next page", "href": "https://example.org/shop/2"},
    {"kind": "field", "label": "Search"},
    {"kind": "button", "label": "Search"},
    {"kind": "button", "label": "Place order"},
    {"kind": "field", "label": "Password", "secret": True},
    {"kind": "list", "label": "Colour"},
]
THINGS = [{"href": "", "secret": False, "risky": False, "many": False} | thing for thing in THINGS]


def did(what):
    with (profile / "did").open("a") as f:
        f.write(what + "\n")


send({"applicationType": "gecko", "marionetteProtocol": 3})
buffer, url = b"", "about:blank"
tab, tabs, shown = "base", {"base": "about:blank"}, 0  # shown: how many times a page has been described
while True:
    while b":" not in buffer or len(buffer.partition(b":")[2]) < int(buffer.partition(b":")[0]):
        piece = connection.recv(65536)
        if not piece:
            sys.exit(0)
        buffer += piece
    size, _, rest = buffer.partition(b":")
    (_, number, name, params), buffer = json.loads(rest[: int(size)]), rest[int(size) :]
    words = "" if "guarded" in url else "thin" if "thin" in url else "the page as the browser built it " * 20
    error, result = None, {"value": None}
    if name == "WebDriver:GetWindowHandle":
        result = {"value": tab}
    elif name == "WebDriver:GetWindowHandles":
        result = list(tabs)
    elif name == "WebDriver:NewWindow":
        result = {"handle": f"tab{len(tabs)}", "type": "tab"}
        tabs[result["handle"]] = "about:blank"
    elif name == "WebDriver:SwitchToWindow":
        tabs[tab] = url
        tab = params["handle"]
        url = tabs[tab]
    elif name in ("WebDriver:ElementClick", "WebDriver:ElementClear", "WebDriver:ElementSendKeys"):
        when, _, which = params["id"].partition("-")
        if int(when) != shown:
            error = {"error": "stale element reference", "message": "The element is no longer attached to the DOM"}
        elif name == "WebDriver:ElementSendKeys":
            did(f"typed into {which}: {params['text']!r}")
            if params["text"].endswith("\ue006"):
                url = "https://example.org/found"
        elif name == "WebDriver:ElementClick":
            did(f"clicked {which}")
            if which == "1":
                url = THINGS[0]["href"]
            elif which == "3" and "trap" in url:
                url = "http://127.0.0.1:9/inside"
    elif name == "WebDriver:ExecuteScript" and "private_ai: view" in params["script"]:
        shown += 1
        text = "\n".join(f'[{n} {t["kind"]} "{t["label"]}"]' for n, t in enumerate(THINGS, 1))
        refs = [{ELEMENT: f"{shown}-{n}"} for n in range(1, len(THINGS) + 1)]
        result = {"value": ["Shop", f"Things for sale at {url}\n{text}\n" + "more goods\n" * params["args"][0] * ("long" in url), refs, THINGS]}
    elif name == "WebDriver:ExecuteScript" and "private_ai: ready" in params["script"]:
        result = {"value": ["complete", 500]}
    elif name == "WebDriver:ExecuteScript" and "private_ai: choose" in params["script"]:
        found = params["args"][1] in ("Red", "Blue")
        if found:
            did(f"chose {params['args'][1]}")
        result = {"value": found}
    elif name == "WebDriver:Navigate":
        url = params["url"]
        with (profile / "opened").open("a") as f:
            f.write(url + "\n")
        if "wanders" in url:
            url = "http://127.0.0.1:9/inside"
        if "slow" in url:
            error = {"error": "timeout", "message": "Navigation timed out after 30000 ms"}
    elif name == "WebDriver:GetCurrentURL":
        result = {"value": url}
    elif name == "WebDriver:ExecuteScript":
        script = params["script"]
        if "outerHTML" in script:
            body = '<iframe src="https://geo.captcha-delivery.example/c"></iframe>' if "guarded" in url else "" if "drawn" in url else words
            result = {"value": f"<html><head><title>Built</title></head><body><p>{body}</p></body></html>"}
        elif "length" in script:
            result = {"value": len(words)}
        else:
            result = {"value": ["Shown", words]}
    send([1, number, error, result])
    if name == "Marionette:Quit":
        sys.exit(0)
