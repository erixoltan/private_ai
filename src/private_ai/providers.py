"""Provider definitions and local configuration.

Every provider is reached through the OpenAI-compatible chat completions
protocol, so one small client covers all of them and users can add any other
compatible endpoint in their config file.

Configuration lives only on this machine:

    ~/.config/private_ai/config.toml     (or $PRIVATE_AI_CONFIG)

    default_provider = "fireworks"       # used by `private-ai chat`; default is local ollama
    FIREWORKS_API_KEY = "fw_..."         # simplest: same names as the environment variables

    [providers.fireworks]
    model = "accounts/fireworks/models/..."
    context_tokens = 100000              # conversation size before older parts are summarised

    [providers.bedrock]
    region = "us-west-2"

    [providers.my_server]                # any OpenAI-compatible endpoint
    base_url = "http://localhost:8080/v1"

An API key in the environment (see ``key_env``) takes precedence over the file.
A provider entry may also set ``api_key`` directly.
"""

from __future__ import annotations

import ipaddress
import os
import stat
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str | None = None  # environment variable holding the API key
    api_key: str | None = None
    model: str | None = None  # None: use the first model the endpoint lists
    # How much conversation to send before the earlier part is replaced by a
    # summary. An estimate, kept well under most models' limits; it also caps
    # what each message costs.
    context_tokens: int = 48_000
    # The provider's own published data-handling policy. This program cannot
    # verify these claims; the URL is here so the user can.
    policy_url: str | None = None
    policy_note: str | None = None

    @property
    def is_local(self) -> bool:
        """True when requests never leave this machine."""
        host = urlsplit(self.base_url).hostname or ""
        if host == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    @property
    def needs_key(self) -> bool:
        return self.key_env is not None and not self.is_local


def _builtin(bedrock_region: str) -> dict[str, Provider]:
    providers = [
        Provider(
            name="ollama",
            base_url=os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434") + "/v1",
            policy_note="Local model: prompts never leave this machine.",
        ),
        Provider(
            name="ollama_cloud",
            base_url="https://ollama.com/v1",
            key_env="OLLAMA_API_KEY",
            policy_url="https://ollama.com/cloud",
            policy_note="Provider states it does not log or retain prompts or responses.",
        ),
        Provider(
            name="fireworks",
            base_url="https://api.fireworks.ai/inference/v1",
            key_env="FIREWORKS_API_KEY",
            # The model list includes routers and models an account may not be able to
            # call, so "first listed" is unreliable here.
            model="accounts/fireworks/models/glm-5p3-flash",
            policy_url="https://docs.fireworks.ai/guides/security_compliance/data_handling",
            policy_note="Provider states it does not store or train on prompts for open models.",
        ),
        Provider(
            name="bedrock",
            base_url=f"https://bedrock-runtime.{bedrock_region}.amazonaws.com/openai/v1",
            key_env="AWS_BEARER_TOKEN_BEDROCK",
            policy_url="https://docs.aws.amazon.com/bedrock/latest/userguide/data-protection.html",
            policy_note="Provider states it does not store prompts or use them for training.",
        ),
    ]
    return {p.name: p for p in providers}


def config_path() -> Path:
    override = os.environ.get("PRIVATE_AI_CONFIG")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "private_ai" / "config.toml"


def config_warnings(path: Path | None = None) -> list[str]:
    """Problems with the config file that the user should know about."""
    path = path or config_path()
    if os.name == "posix" and path.exists():
        if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            return [f"{path} is readable by other users; run: chmod 600 {path}"]
    return []


def default_provider_name(path: Path | None = None) -> str:
    path = path or config_path()
    if path.exists():
        with path.open("rb") as f:
            return tomllib.load(f).get("default_provider", "ollama")
    return "ollama"


def load_providers(path: Path | None = None) -> dict[str, Provider]:
    """Built-in providers merged with the user's config file and environment."""
    path = path or config_path()
    config: dict = {}
    if path.exists():
        with path.open("rb") as f:
            config = tomllib.load(f)
    settings: dict[str, dict] = config.get("providers", {})

    region = (
        settings.get("bedrock", {}).get("region")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-west-2"
    )
    providers = _builtin(region)

    for name, entry in settings.items():
        fields = {k: entry[k] for k in ("base_url", "key_env", "api_key", "model", "context_tokens") if k in entry}
        if name in providers:
            providers[name] = replace(providers[name], **fields)
        elif "base_url" in fields:
            providers[name] = Provider(name=name, **fields)
        else:
            raise ValueError(f"{path}: provider '{name}' needs a base_url")

    for name, p in providers.items():
        # Key lookup order: environment, top-level NAME = "..." in the file, then api_key.
        named_key = (os.environ.get(p.key_env) or config.get(p.key_env)) if p.key_env else None
        providers[name] = replace(p, base_url=p.base_url.rstrip("/"), api_key=named_key or p.api_key)
    return providers
