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

API keys and settings go in `~/.config/private_ai/config.toml`:

    default_provider = "fireworks"      # without this, chat uses local Ollama
    FIREWORKS_API_KEY = "..."
    OLLAMA_API_KEY = "..."

Conversations and the agent's memory of your preferences are plain files in
`~/.local/share/private_ai/`. Delete or edit them whenever you like.

The agent can read and write local files, but only after asking you. The
question says whether the contents would stay on your machine or go to a cloud
provider. The folder holding your API keys is always off limits.
