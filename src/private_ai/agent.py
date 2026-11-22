"""The generic agent: one conversation between the user and a model, with tools.

This module has no knowledge of terminals or browsers. An interface passes in
a UI object (see tools.UI) and calls send() for each thing the user says.
"""

from __future__ import annotations

import os
import time

from private_ai.client import ProviderError, chat_stream
from private_ai.providers import Provider
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
        return {"role": "system", "content": prompt}

    def _complete(self, messages: list[dict]) -> dict:
        try:
            return chat_stream(
                self.provider, messages, self.model, TOOLS if self.use_tools else None, self.ui.text
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
                reply = self._complete([self._system_message(), *messages])
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
        self.store.save(self.conversation)
