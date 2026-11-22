"""Tests run against a fake provider on localhost; nothing leaves the machine."""

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from private_ai.client import ProviderError, chat, list_models, local_models
from private_ai.providers import Provider, load_providers


class FakeProvider(BaseHTTPRequestHandler):
    seen_auth = None

    def _send(self, code, payload, headers=()):
        body = json.dumps(payload).encode()
        self.send_response(code)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        FakeProvider.seen_auth = self.headers.get("Authorization")
        if self.path == "/v1/models":
            self._send(200, {"data": [{"id": "m1"}, {"id": "m2"}]})
        elif self.path == "/api/tags":
            self._send(200, {"models": [{"name": "m1"}, {"name": "m2:latest", "remote_host": "https://x"}]})
        elif self.path == "/redirect/models":
            self._send(302, {}, [("Location", f"http://127.0.0.1:{self.server.server_port}/v1/models")])
        else:
            self._send(401, {"error": "bad key"})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": "pong " + request["model"]}}]})

    def log_message(self, *args):
        pass


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeProvider)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.root = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def provider(self, path="/v1", **kw):
        return Provider(name="fake", base_url=self.root + path, **kw)

    def test_list_models_sends_key(self):
        self.assertEqual(list_models(self.provider(api_key="secret")), ["m1", "m2"])
        self.assertEqual(FakeProvider.seen_auth, "Bearer secret")

    def test_chat(self):
        msg = chat(self.provider(model="m1"), [{"role": "user", "content": "ping"}])
        self.assertEqual(msg["content"], "pong m1")

    def test_relayed_model_refused_on_local_provider(self):
        for name in ("m2", "m2:latest"):
            with self.assertRaisesRegex(ProviderError, "not a local model"):
                chat(self.provider(), [], model=name)
        self.assertEqual(local_models(self.provider()), ["m1"])

    def test_chat_requires_model(self):
        with self.assertRaises(ProviderError):
            chat(self.provider(), [])

    def test_http_error_is_reported(self):
        with self.assertRaisesRegex(ProviderError, "HTTP 401"):
            list_models(self.provider(path="/nope"))

    def test_redirect_refused(self):
        with self.assertRaisesRegex(ProviderError, "refused redirect"):
            list_models(self.provider(path="/redirect", api_key="secret"))

    def test_key_not_sent_over_plain_http_to_remote_host(self):
        remote = Provider(name="r", base_url="http://example.invalid/v1", api_key="secret")
        with self.assertRaisesRegex(ProviderError, "unencrypted"):
            list_models(remote)


class ConfigTest(unittest.TestCase):
    def test_is_local(self):
        self.assertTrue(Provider("a", "http://localhost:11434/v1").is_local)
        self.assertTrue(Provider("a", "http://[::1]:8080/v1").is_local)
        self.assertFalse(Provider("a", "https://ollama.com/v1").is_local)

    def test_config_file_overrides_and_adds(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.toml"
            path.write_text(
                'FIREWORKS_API_KEY = "k"\n'
                '[providers.fireworks]\nmodel = "x"\n'
                '[providers.bedrock]\nregion = "eu-west-1"\n'
                '[providers.mine]\nbase_url = "http://localhost:9/v1/"\n'
            )
            providers = load_providers(path)
        self.assertEqual(providers["fireworks"].model, "x")
        if "FIREWORKS_API_KEY" not in os.environ:
            self.assertEqual(providers["fireworks"].api_key, "k")
        self.assertIn("eu-west-1", providers["bedrock"].base_url)
        self.assertEqual(providers["mine"].base_url, "http://localhost:9/v1")


if __name__ == "__main__":
    unittest.main()
