"""Stands in for Firefox in tests: speaks enough of its remote-control protocol
to be started, asked for pages and told to quit. What a page holds depends on
words in its address: "thin" has almost no text, "slow" fails to finish
loading, "drawn" has its text only on screen and not in its HTML, and
"wanders" ends up at an address on this machine, and "guarded" shows a
captcha in place of the page."""

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


send({"applicationType": "gecko", "marionetteProtocol": 3})
buffer, url = b"", "about:blank"
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
    if name == "WebDriver:Navigate":
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
