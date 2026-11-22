"""Minimal OpenAI-compatible HTTP client built on the standard library.

Privacy properties this module is responsible for:

* It talks only to the provider's configured base URL. Redirects are refused,
  so an API key can never be forwarded to a different host.
* API keys are sent only over HTTPS, or to this machine itself.
* A provider on this machine is only allowed to run models that really are
  local. Ollama can hold "cloud" models that it silently relays to ollama.com;
  those are refused here, so "local" always means local.
* Nothing is sent except what the caller passes in: no telemetry, no
  identifying headers.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from private_ai import __version__
from private_ai.providers import Provider


class ProviderError(Exception):
    """A request to a provider failed. The message is safe to show the user."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError(f"refused redirect ({code}) to {newurl}")


_opener = urllib.request.build_opener(_NoRedirect)


def _request(
    provider: Provider, path: str, body: dict | None = None, timeout: float = 30, url: str | None = None
) -> dict:
    url = url or provider.base_url + path
    headers = {"Accept": "application/json", "User-Agent": f"private_ai/{__version__}"}
    if provider.api_key:
        if not (url.startswith("https://") or provider.is_local):
            raise ProviderError("refusing to send an API key over unencrypted HTTP")
        headers["Authorization"] = f"Bearer {provider.api_key}"
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with _opener.open(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read(2000).decode(errors="replace").strip()
        raise ProviderError(f"HTTP {e.code} from {url}: {detail or e.reason}") from None
    except urllib.error.URLError as e:
        raise ProviderError(f"cannot reach {url}: {e.reason}") from None
    except TimeoutError:
        raise ProviderError(f"timed out after {timeout:g}s: {url}") from None
    except json.JSONDecodeError:
        raise ProviderError(f"non-JSON response from {url}") from None


def list_models(provider: Provider, timeout: float = 30) -> list[str]:
    data = _request(provider, "/models", timeout=timeout)
    return [m["id"] for m in data.get("data", [])]


def relayed_models(provider: Provider, timeout: float = 30) -> set[str]:
    """Models on a local Ollama server that are actually run on a remote host.

    Uses Ollama's native model list, which is the only place this is reported.
    Returns an empty set for servers that are not Ollama.
    """
    url = provider.base_url.removesuffix("/v1") + "/api/tags"
    try:
        data = _request(provider, "", timeout=timeout, url=url)
    except ProviderError:
        return set()
    return {m["name"] for m in data.get("models", []) if m.get("remote_host")}


def _is_relayed(model: str, relayed: set[str]) -> bool:
    return model in relayed or (":" not in model and f"{model}:latest" in relayed)


def local_models(provider: Provider, timeout: float = 30) -> list[str]:
    """Like list_models, but without models a local server relays elsewhere."""
    models = list_models(provider, timeout)
    if provider.is_local:
        relayed = relayed_models(provider, timeout)
        models = [m for m in models if not _is_relayed(m, relayed)]
    return models


def chat(
    provider: Provider,
    messages: list[dict],
    model: str | None = None,
    max_tokens: int | None = None,
    timeout: float = 60,
) -> dict:
    """Send a chat completion and return the assistant message dict."""
    model = model or provider.model
    if not model:
        raise ProviderError(f"no model configured for provider '{provider.name}'")
    if provider.is_local and _is_relayed(model, relayed_models(provider, timeout)):
        raise ProviderError(
            f"'{model}' is not a local model: {provider.name} would relay it to a remote host. "
            "Use a cloud provider entry for it instead."
        )
    body: dict = {"model": model, "messages": messages, "stream": False}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    data = _request(provider, "/chat/completions", body, timeout=timeout)
    try:
        return data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise ProviderError(f"unexpected response shape: {json.dumps(data)[:300]}") from None
