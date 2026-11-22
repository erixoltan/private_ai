"""The generic agent: one conversation between the user and a model, with tools.

This module has no knowledge of terminals or browsers. An interface passes in
a UI object (see tools.UI) and calls send() for each thing the user says.
"""

from __future__ import annotations

import json
import os
import re
import time

from private_ai.client import ProviderError, chat, chat_stream, check_model, local_models
from private_ai.providers import Provider, config_path
from private_ai.store import Store
from private_ai.tools import TOOLS, UI, Toolbox

MAX_TOOL_ROUNDS = 25

SYSTEM_PROMPT = """\
You are a private, general-purpose assistant running on the user's own computer.
Conversations and memory are stored only on this computer.

Today is {date}. The current folder is {cwd}.

You have tools for long-term memory, for searching earlier conversations, and
for reading and writing local files. The user is asked before any file is
touched, so just call the tool rather than asking first in chat.
When the user tells you a lasting preference or fact about themselves, save it
with `remember` without being asked. Do not save passing details.
When the user refers to something from an earlier conversation that you cannot
see, use `search_conversations`.

What you remember about the user:
{memory}
"""

SUMMARY_NOTE = """
This conversation is long. Its earlier part is no longer shown to you; these
are your notes on it:
{summary}
"""

SUMMARY_PROMPT = """You are compacting the earlier part of a long conversation between a user and
an assistant so that it can continue without it. Write notes that preserve
everything needed to carry on: what the user wants, facts about the user and
their situation, decisions made, conclusions reached, open questions, and exact
names, numbers, dates and file paths. Say what was happening most recently.
Write notes, not a reply to the user. Stay under 700 words."""

# The model picks a colour by name from this list; the interface decides what
# each name looks like, so it stays readable in light and dark themes.
PALETTE = ("rose", "red", "orange", "amber", "green", "teal", "blue", "indigo", "purple", "gray")

IDENTITY_PROMPT = f"""Give a conversation a look, so its owner can tell it apart from their other
conversations at a glance. Reply with only a JSON object:
{{"title": "...", "emoji": "...", "color": "..."}}
title: at most 6 words, specific to the subject, no quotes or final punctuation.
emoji: exactly one emoji that suits the subject.
color: one of {", ".join(PALETTE)}. Choose by subject and mood; use gray for
technical or neutral subjects."""


def estimate_tokens(messages: list[dict]) -> int:
    """Rough size of messages as a model counts them. Errs on the high side."""
    chars = 0
    for m in messages:
        chars += len(m.get("content") or "") + 20
        for call in m.get("tool_calls") or []:
            chars += len(call["function"]["name"]) + len(call["function"]["arguments"]) + 20
    return int(chars / 3.2)


def _transcript(messages: list[dict], limit: int) -> str:
    """Messages as plain text for the model to read, long ones cut short."""
    lines = []
    for m in messages:
        content = (m.get("content") or "").strip()
        if len(content) > limit:
            content = content[:limit] + " [...]"
        if m["role"] == "user":
            lines.append(f"User: {content}")
        elif m["role"] == "assistant":
            for call in m.get("tool_calls") or []:
                lines.append(f"(assistant used {call['function']['name']}: {call['function']['arguments'][:300]})")
            if content:
                lines.append(f"Assistant: {content}")
        elif m["role"] == "tool":
            lines.append(f"(result: {content[: limit // 4]})")
    return "\n\n".join(lines)


def ready(provider: Provider) -> Provider:
    """The provider, or an error explaining which key is missing."""
    if provider.needs_key and not provider.api_key:
        raise ProviderError(f"no API key for {provider.name}: set {provider.key_env} or add it to {config_path()}")
    return provider


def pick_model(provider: Provider, model: str | None) -> str:
    """The requested model, else the provider's default, else the first one it lists."""
    model = model or provider.model
    if model:
        return check_model(provider, model)  # fail now, not after the user's first message
    models = local_models(provider)
    if not models:
        raise ProviderError(f"{provider.name} lists no models; choose one explicitly")
    return models[0]


class Agent:
    def __init__(self, provider: Provider, model: str, store: Store, ui: UI, conversation: dict | None = None):
        self.provider = provider
        self.model = model
        self.store = store
        self.ui = ui
        self.toolbox = Toolbox(store, ui, provider)
        self.use_tools = True
        self.conversation = conversation or store.new_conversation()
        self.toolbox.conversation_id = self.conversation["id"]

    def _system_message(self) -> dict:
        facts = self.store.memory()
        prompt = SYSTEM_PROMPT.format(
            date=time.strftime("%A, %B %-d, %Y"),
            cwd=os.getcwd(),
            memory="\n".join(f"- {f}" for f in facts) or "(nothing yet)",
        )
        if self.conversation.get("summary"):
            prompt += SUMMARY_NOTE.format(summary=self.conversation["summary"])
        return {"role": "system", "content": prompt}

    # ---- long conversations ----
    #
    # The file on disk always keeps every message. What changes is how much is
    # sent: once the part being sent passes the provider's context_tokens, the
    # oldest messages are replaced by a summary ("summary" covers the first
    # "summary_upto" messages).

    def _visible(self) -> list[dict]:
        return self.conversation["messages"][self.conversation.get("summary_upto", 0) :]

    def _compact_if_needed(self) -> None:
        budget = self.provider.context_tokens
        if estimate_tokens(self._visible()) <= budget:
            return
        messages = self.conversation["messages"]
        start = self.conversation.get("summary_upto", 0)
        # Cut only where the user speaks, so a tool call is never separated from its result.
        cuts = [i for i in range(start + 1, len(messages)) if messages[i]["role"] == "user"]
        if not cuts:
            return  # one enormous exchange; nothing older to fold away
        cut = next((i for i in cuts if estimate_tokens(messages[i:]) <= budget // 2), cuts[-1])
        self.ui.notice("this conversation is long; summarising the earlier part for the model")
        earlier = _transcript(messages[start:cut], 6000)
        if self.conversation.get("summary"):
            earlier = f"Notes on the part before this:\n{self.conversation['summary']}\n\n{earlier}"
        try:
            reply = chat(
                self.provider,
                [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": earlier}],
                self.model,
                max_tokens=4000,
                timeout=180,
            )
        except ProviderError as e:
            self.ui.notice(f"could not summarise ({e}); sending the whole conversation")
            return
        summary = (reply.get("content") or "").strip()
        if summary:
            self.conversation["summary"], self.conversation["summary_upto"] = summary, cut

    # ---- title, emoji and colour ----

    def _describe(self) -> None:
        """Ask the model, once, for a title, emoji and colour for this conversation."""
        conv = self.conversation
        if conv.get("emoji") or conv.get("identity_tries", 0) >= 2:
            return
        conv["identity_tries"] = conv.get("identity_tries", 0) + 1
        opening = _transcript([m for m in conv["messages"] if m["role"] in ("user", "assistant")][:4], 1500)
        try:
            reply = chat(
                self.provider,
                [{"role": "system", "content": IDENTITY_PROMPT}, {"role": "user", "content": opening}],
                self.model,
                max_tokens=1000,
                timeout=60,
            )
            found = json.loads(re.search(r"\{.*\}", reply.get("content") or "", re.S).group())
            title = " ".join(str(found["title"]).split()).strip("\"'.")
            emoji = str(found["emoji"]).strip()
            color = str(found["color"]).strip().lower()
            if not title or not emoji or len(emoji) > 8 or any(c.isascii() for c in emoji) or color not in PALETTE:
                raise ValueError
        except (ProviderError, ValueError, KeyError, AttributeError, TypeError):
            pass  # cosmetic; it gets one more chance after the next message
        else:
            conv["title"], conv["emoji"], conv["color"] = title[:60], emoji, color
        self.store.save(conv)

    def _complete(self, messages: list[dict]) -> dict:
        # Saved reasoning is for the user to read; it is never sent back to a provider.
        messages = [{k: v for k, v in m.items() if k != "thinking"} for m in messages]
        try:
            return chat_stream(
                self.provider,
                messages,
                self.model,
                TOOLS if self.use_tools else None,
                self.ui.text,
                self.ui.thinking,
            )
        except ProviderError as e:
            # Some local models cannot call tools at all; carry on as plain chat.
            if self.use_tools and "tool" in str(e).lower() and "HTTP 400" in str(e):
                self.use_tools = False
                self.ui.notice(f"{self.model} cannot use tools; memory and file access are off for this session")
                return self._complete(messages)
            raise

    def send(self, text: str) -> None:
        """Handle one user message: stream the reply, running tools as the model asks."""
        messages = self.conversation["messages"]
        turn_start = len(messages)
        messages.append({"role": "user", "content": text})
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                self._compact_if_needed()
                reply = self._complete([self._system_message(), *self._visible()])
                messages.append(reply)
                calls = reply.get("tool_calls")
                if not calls:
                    break
                for call in calls:
                    result = self.toolbox.run(call["function"]["name"], call["function"]["arguments"])
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            else:
                self.ui.notice(f"stopped after {MAX_TOOL_ROUNDS} tool rounds")
        except BaseException:
            # Failed or interrupted: drop the half-finished turn so history stays valid.
            del messages[turn_start:]
            raise
        self.conversation["provider"] = self.provider.name
        self.conversation["model"] = self.model
        self.store.save(self.conversation)
        self._describe()
