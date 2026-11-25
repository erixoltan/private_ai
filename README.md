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
    BEDROCK_API_KEY = "..."

    [providers.bedrock]
    region = "us-east-2"                # the Region the Bedrock key was created in

Conversations, the agent's memory of your preferences, pages it has read and
research reports are plain files in `~/.local/share/private_ai/`. Delete or edit them whenever you like.

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

## The web

The agent can search the web, read your news feeds and read web pages.

    private-ai providers                   # also lists the search services
    private-ai search "a query"            # try the search service chat would use
    private-ai search "a query" --with wikipedia

No single search service is built in. Choose one with `default_search` in the
config file, with `/search NAME` in the terminal, or in the web page:

| Name | What it is | Needs |
|---|---|---|
| `searxng` | A SearXNG server you run yourself. It asks public search engines on your behalf, with no account. | a running server |
| `brave` | Brave Search API | `BRAVE_API_KEY` |
| `ollama` | Ollama's web search API | `OLLAMA_API_KEY` |
| `wikipedia` | Wikipedia only | nothing |

Without `default_search`, the first of these that is set up is used, and
Wikipedia if none is.

What leaves your machine, and when you are asked:

* **Searching.** A query goes to the search service, which your model
  provider's privacy policy does not cover. You are shown each query and where
  it is going before it is sent. You can allow one search or all searches for
  the session.
* **News.** The feeds are downloaded whole and filtered on your machine, so
  the news sites never learn what you asked about. You are not asked.
* **Reading a page.** The site sees a request from your address, with no
  cookies. A page is read without asking only if its address came from you or
  from a search or news result. For any other address, which the model made
  up, you are asked first.
* **Pages that refuse.** Many sites block programs, and pages built by
  JavaScript come back empty. If you have an Ollama key you are offered
  Ollama's page-reading service, which would see the address. Otherwise the
  agent uses the excerpt from the search result and says so.
* **Limits.** At most 6 searches and 8 pages per message.

Every page read is saved in `~/.local/share/private_ai/pages/`, so the agent
can find it again in later conversations. Text from the web is marked as
untrusted before the model sees it, but no marking makes a model immune to a
page written to mislead it: file writes still ask you every time.

To run SearXNG on this machine only, with Docker:

    mkdir -p ~/.config/searxng
    cat > ~/.config/searxng/settings.yml <<EOF
    use_default_settings: true
    server:
      secret_key: "$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
      limiter: false
    search:
      formats: [html, json]
    EOF
    docker run -d --name searxng --restart unless-stopped -p 127.0.0.1:8888:8080 \
      -v ~/.config/searxng/settings.yml:/etc/searxng/settings.yml:ro searxng/searxng

Then in `~/.config/private_ai/config.toml`:

    default_search = "searxng"

    [news]                              # optional: your own feeds instead of the built-in four
    feeds = ["https://feeds.bbci.co.uk/news/world/rss.xml"]

## Research

The research agent takes a topic and works through five steps:

1. Plan the search queries.
2. Search, and read the pages, taking notes on each.
3. Look for what is missing, then search and read again.
4. Plan the report. The model thinks through what the notes establish and
   writes an outline: for each section, the points to make, the sources for
   each, and its reasoning. The plan is kept, and you can read it.
5. Write the report from the plan and the notes. This pass asks the model for
   little thinking, since the thinking is in the plan.

    private-ai research "How do heat pumps perform below freezing?"
    private-ai research --list
    private-ai research --show 1 > report.md
    private-ai research --plan 1           # the outline and reasoning behind report 1
    private-ai research --retry 1          # write the report again from saved notes

In the terminal chat use `/research TOPIC`. In the web page, research is
listed in the side panel under the conversations: "New research" starts a job,
which runs in the background while you carry on, for as long as
`private-ai web` is running, and clicking one opens its report. Like a
conversation, each piece of research gets a short title, an emoji and a colour
from the model, which is shown only the topic for this. You can also just ask
for research in a conversation: the assistant proposes a topic and you approve
it.

* **One approval, stated limits.** Starting a job allows up to 8 searches with
  the current search service and up to 16 pages, without a question for each
  (`--searches` and `--pages` change the limits on the command line).
* **The researcher sees only the topic.** It is not given your memory,
  conversations or files, so nothing private can end up in a search query
  unless you wrote it in the topic.
* **Pages that refuse are skipped**, and listed in the saved job. No
  third-party reading service is used.
* **Reports are kept** in `~/.local/share/private_ai/research/`, with the notes
  and sources behind them. The assistant sees the list of reports in every
  conversation and can read them, so you can discuss one later.

A report is only as good as the pages found and the model that read them.
Check the sources before relying on a claim.

## Files

The agent can read and write local files, but only after asking you. The
question says whether the contents would stay on your machine or go to a cloud
provider. Before a file is written you are shown what would be written: the
contents of a new file, or the lines that change in an existing one.

* The folder holding your API keys is always off limits.
* The assistant cannot write to its own data folder with the file tools.
* Places that usually hold secrets (`~/.ssh`, `~/.aws`, browser profiles,
  `.env` files, private keys and the like) are asked about every time, with a
  warning, even inside a folder you approved for the session.
* Writing a file the system runs by itself (`~/.bashrc`, autostart entries,
  git hooks, system folders) also carries a warning.

These lists cover well-known locations and cannot be complete. Read the path
in every question.
