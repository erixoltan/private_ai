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

import contextlib
import http.client
import json
import re
import urllib.error
import urllib.request

from collections.abc import Callable

from private_ai import __version__
from private_ai.providers import Provider


class ProviderError(Exception):
    """A request to a provider failed. The message is safe to show the user."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError(f"refused redirect ({code}) to {newurl}")


_opener = urllib.request.build_opener(_NoRedirect)


@contextlib.contextmanager
def _open(provider, url: str, body: dict | None, timeout: float, key_header: str = "Authorization"):
    """Open a request and translate every failure, including ones while reading.

    provider is anything with api_key and is_local: a model provider or a search backend."""
    headers = {"Accept": "application/json", "User-Agent": f"private_ai/{__version__}"}
    if provider.api_key:
        if not (url.startswith("https://") or provider.is_local):
            raise ProviderError("refusing to send an API key over unencrypted HTTP")
        headers[key_header] = f"Bearer {provider.api_key}" if key_header == "Authorization" else provider.api_key
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with _opener.open(req, timeout=timeout) as resp:
            yield resp
    except urllib.error.HTTPError as e:
        detail = e.read(2000).decode(errors="replace").strip()
        e.close()
        raise ProviderError(f"HTTP {e.code} from {url}: {detail or e.reason}") from None
    except urllib.error.URLError as e:
        raise ProviderError(f"cannot reach {url}: {e.reason}") from None
    except TimeoutError:
        raise ProviderError(f"timed out after {timeout:g}s: {url}") from None
    except json.JSONDecodeError:
        raise ProviderError(f"non-JSON response from {url}") from None
    except (ConnectionError, http.client.HTTPException) as e:
        raise ProviderError(f"connection to {url} failed: {e}") from None


def _request(
    provider,
    path: str,
    body: dict | None = None,
    timeout: float = 30,
    url: str | None = None,
    key_header: str = "Authorization",
) -> dict:
    with _open(provider, url or provider.base_url + path, body, timeout, key_header) as resp:
        return json.load(resp)


def list_models(provider: Provider, timeout: float = 30) -> list[str]:
    bedrock = _BEDROCK_RUNTIME.match(provider.base_url)
    if bedrock:
        return _bedrock_models(provider, f"https://bedrock.{bedrock[1]}.amazonaws.com", timeout)
    data = _request(provider, "/models", timeout=timeout)
    return [m["id"] for m in data.get("data", [])]


# Amazon Bedrock's runtime endpoint has no model list; its management endpoint does.
_BEDROCK_RUNTIME = re.compile(r"https://bedrock-runtime\.([a-z0-9-]+)\.amazonaws\.com/")


def _bedrock_models(provider: Provider, management_url: str, timeout: float) -> list[str]:
    foundation = _request(provider, "", timeout=timeout, url=management_url + "/foundation-models?byOutputModality=TEXT")
    profiles = _request(provider, "", timeout=timeout, url=management_url + "/inference-profiles?maxResults=1000")
    return bedrock_model_ids(foundation, profiles)


def bedrock_model_ids(foundation: dict, profiles: dict) -> list[str]:
    """The names a Bedrock request can use: models that run in the account's
    own Region, then cross-Region inference profiles for text models.

    Global profiles are left out: they may process a request in any Region in
    the world. A profile tied to a geography (us., eu., ...) stays inside it.
    A global profile still works if the user types its name."""
    summaries = foundation.get("modelSummaries", [])
    ids = [m["modelId"] for m in summaries if "ON_DEMAND" in m.get("inferenceTypesSupported", [])]
    text_models = {m["modelId"] for m in summaries}
    for profile in profiles.get("inferenceProfileSummaries", []):
        name = profile["inferenceProfileId"]
        runs = {m.get("modelArn", "").rpartition("/")[2] for m in profile.get("models", [])}
        if not name.startswith("global.") and runs & text_models:
            ids.append(name)
    return ids


def relayed_models(provider: Provider, timeout: float = 30) -> dict[str, str]:
    """Models on a local Ollama server that are actually run on a remote host,
    mapped to the name the remote host knows them by.

    Uses Ollama's native model list, which is the only place this is reported.
    Returns nothing for servers that are not Ollama.
    """
    url = provider.base_url.removesuffix("/v1") + "/api/tags"
    try:
        data = _request(provider, "", timeout=timeout, url=url)
    except ProviderError:
        return {}
    return {m["name"]: m.get("remote_model") or m["name"] for m in data.get("models", []) if m.get("remote_host")}


def _relayed_as(model: str, relayed: dict[str, str]) -> str | None:
    return relayed.get(model) or (relayed.get(f"{model}:latest") if ":" not in model else None)


def local_models(provider: Provider, timeout: float = 30) -> list[str]:
    """Like list_models, but without models a local server relays elsewhere."""
    models = list_models(provider, timeout)
    if provider.is_local:
        relayed = relayed_models(provider, timeout)
        models = [m for m in models if not _relayed_as(m, relayed)]
    return models


def check_model(provider: Provider, model: str | None, timeout: float = 30) -> str:
    """The model to use, after confirming a local provider would really run it locally."""
    model = model or provider.model
    if not model:
        raise ProviderError(f"no model configured for provider '{provider.name}'")
    remote_name = _relayed_as(model, relayed_models(provider, timeout)) if provider.is_local else None
    if remote_name:
        raise ProviderError(
            f"'{model}' is a cloud model, but '{provider.name}' is the provider for models that run on this "
            f"machine, and it only runs those. Pick a cloud provider instead; for Ollama's cloud that is: "
            f"--provider ollama_cloud --model {remote_name}"
        )
    return model


def chat(
    provider: Provider,
    messages: list[dict],
    model: str | None = None,
    max_tokens: int | None = None,
    timeout: float = 60,
    reasoning_effort: str | None = None,
) -> dict:
    """Send a chat completion and return the assistant message dict, with the
    reason the model stopped under "finish_reason" ("length": it ran out of max_tokens).

    reasoning_effort ("low" or "high") asks a thinking model to think less or
    more. It is a request: a provider or model that rejects it is asked again without it."""
    model = check_model(provider, model, timeout)
    body: dict = {"model": model, "messages": messages, "stream": False}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    try:
        asked = {**body, "reasoning_effort": reasoning_effort}
        data = _request(provider, "/chat/completions", asked, timeout=timeout) if reasoning_effort else None
    except ProviderError as e:
        if "HTTP 400" not in str(e) and "HTTP 422" not in str(e):
            raise
        data = None
    data = data or _request(provider, "/chat/completions", body, timeout=timeout)
    try:
        return {**data["choices"][0]["message"], "finish_reason": data["choices"][0].get("finish_reason")}
    except (KeyError, IndexError, TypeError):
        raise ProviderError(f"unexpected response shape: {json.dumps(data)[:300]}") from None


def chat_stream(
    provider: Provider,
    messages: list[dict],
    model: str | None = None,
    tools: list[dict] | None = None,
    on_text: Callable[[str], None] = lambda delta: None,
    on_thinking: Callable[[str], None] = lambda delta: None,
    timeout: float = 300,
) -> dict:
    """Stream a chat completion. Calls on_text for each piece of the reply and
    on_thinking for each piece of the model's reasoning, and returns the
    complete assistant message, including any tool calls.

    Reasoning is returned under the non-standard key "thinking". It is for
    display only: remove it before sending the message back to a provider."""
    model = check_model(provider, model, timeout)
    body: dict = {"model": model, "messages": messages, "stream": True}
    if tools:
        body["tools"] = tools

    text: list[str] = []
    thinking: list[str] = []
    calls: list[dict] = []
    slots: dict[int, dict] = {}  # stream index -> call being assembled
    with _open(provider, provider.base_url + "/chat/completions", body, timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("error"):
                raise ProviderError(f"provider error mid-stream: {json.dumps(chunk['error'])[:300]}")
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    thinking.append(reasoning)
                    on_thinking(reasoning)
                if delta.get("content"):
                    text.append(delta["content"])
                    on_text(delta["content"])
                for part in delta.get("tool_calls") or []:
                    index = part.get("index", 0)
                    call = slots.get(index)
                    # Some servers reuse index 0 for every call; a new id means a new call.
                    if call is None or (part.get("id") and call["id"] and part["id"] != call["id"]):
                        call = slots[index] = {"id": "", "name": "", "arguments": ""}
                        calls.append(call)
                    fn = part.get("function") or {}
                    call["id"] = call["id"] or part.get("id") or ""
                    call["name"] += fn.get("name") or ""
                    call["arguments"] += fn.get("arguments") or ""

    message: dict = {"role": "assistant", "content": "".join(text)}
    if thinking:
        message["thinking"] = "".join(thinking)
    if calls:
        message["tool_calls"] = [
            {
                "id": c["id"] or f"call_{i}",
                "type": "function",
                "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
            }
            for i, c in enumerate(calls)
        ]
    return message
