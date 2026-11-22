# private_ai

by Erik Zoltan

Personal use AI with local privacy enforced.

This AI agent program enforces a local privacy policy.

* Your information is stored on your computer.

* Local AI models can be used that never send your data anywhere.

* Cloud AI models can be used if they have a policy of no training and no data retention.

* You can configure a local privacy policy that protects your information.

## Using it

Requires Python 3.11 or later. There are no other dependencies.

    uv pip install -e .        # or: pip install -e .

    private-ai providers       # what is configured, and each provider's data policy
    private-ai check           # test the connection to each provider
    private-ai chat            # talk to the agent in the terminal
    private-ai chat --provider fireworks
    private-ai chat --resume   # continue the latest conversation
    private-ai web             # the same agent in a web page, on this machine only

API keys and settings go in `~/.config/private_ai/config.toml`:

    default_provider = "fireworks"      # without this, chat uses local Ollama
    FIREWORKS_API_KEY = "..."
    OLLAMA_API_KEY = "..."

Conversations and the agent's memory of your preferences are plain files in
`~/.local/share/private_ai/`. Delete or edit them whenever you like.

`private-ai web` prints a link containing a one-time secret. The page is served
only on 127.0.0.1, only to a browser that opened that link, and only until you
press Ctrl-C. It loads nothing from the internet. To use it from another
computer, forward the port over SSH (`ssh -L 8765:127.0.0.1:8765 host`) rather
than exposing it. If you close the tab while a reply is being written, the
reply is finished and saved, provided it completes within two minutes
(`--detached-timeout`).

After the first reply, the model gives each conversation a short title, an
emoji and a colour, so conversations are easy to tell apart. Long conversations
keep working: once one passes a size limit (`context_tokens`, per provider,
default 48000), the model is sent a summary of the earlier part instead of the
whole thing. Nothing is removed from the saved conversation.

If you ask something in the wrong conversation, move it: in the web page use
the arrows button on your question, in the terminal `/move N` (or `/move new`).
The question and its answer go to the end of the conversation you choose.

The agent can read and write local files, but only after asking you. The
question says whether the contents would stay on your machine or go to a cloud
provider. The folder holding your API keys is always off limits.
