"""A title, emoji and colour for a conversation or a piece of research, chosen
by the model so that its owner can tell it apart from the others at a glance."""

from __future__ import annotations

import json
import re

from private_ai.client import ProviderError, chat
from private_ai.providers import Provider

# The model picks a colour by name from this list; the interface decides what
# each name looks like, so it stays readable in light and dark themes.
PALETTE = ("rose", "red", "orange", "amber", "green", "teal", "blue", "indigo", "purple", "gray")

IDENTITY_PROMPT = f"""Give a {{what}} a look, so its owner can tell it apart from their other
{{what}}s at a glance. Reply with only a JSON object:
{{{{"title": "...", "emoji": "...", "color": "..."}}}}
title: at most 6 words, specific to the subject, no quotes or final punctuation.
emoji: exactly one emoji that suits the subject.
color: one of {", ".join(PALETTE)}. Choose by subject and mood; use gray for
technical or neutral subjects."""


def choose(provider: Provider, model: str, text: str, what: str = "conversation") -> dict | None:
    """{"title", "emoji", "color"} for the text, or None when the model gives nothing usable.
    Cosmetic, so every failure is quiet."""
    try:
        reply = chat(
            provider,
            [{"role": "system", "content": IDENTITY_PROMPT.format(what=what)}, {"role": "user", "content": text}],
            model,
            max_tokens=1000,
            timeout=60,
        )
        found = json.loads(re.search(r"\{.*\}", reply.get("content") or "", re.S).group())
        title = " ".join(str(found["title"]).split()).strip("\"'.")
        emoji = str(found["emoji"]).strip()
        color = str(found["color"]).strip().lower()
        if not title or not emoji or len(emoji) > 8 or any(c.isascii() for c in emoji) or color not in PALETTE:
            return None
    except (ProviderError, ValueError, KeyError, AttributeError, TypeError):
        return None
    return {"title": title[:60], "emoji": emoji, "color": color}
