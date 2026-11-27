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
than exposing it.

Work carries on in the background for as long as `private-ai web` is running,
whether or not the page is open:

* A reply that is being written is finished and saved. To stop replies that
  nobody is watching after a while, start with `--detached-timeout SECONDS`.
* A permission question waits for you. Open questions are listed at the top
  of the side panel under "Waiting for you", and counted in the tab's title.
  Nothing is allowed on your behalf while you are away.
* Research keeps going, as described below.

Above the message box the page says what is going on: waiting for the model,
the model thinking or writing, a tool or command running, waiting for your
answer, or idle, with how long that has lasted. If the model was producing
words and then stopped, it also says for how long nothing new has arrived,
which is how slow is told from stuck. The button beside it is blue when there
is nothing to send, green when it would send what you typed, and red when it
would stop the work. Stop takes effect at once, even while the model is silent.

Pressing Ctrl-C stops all of it. A reply that was not finished is lost.
Research keeps what it had gathered, and is shown as stopped or interrupted.

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
  JavaScript come back empty. Such a page is opened in the assistant's own
  browser instead (see The browser). If that fails too and you have an Ollama
  key, you are offered Ollama's page-reading service, which would see the
  address. Otherwise the agent uses the excerpt from the search result and
  says so.
* **Limits.** At most 6 searches and 8 pages per message.

Every page read is saved in `~/.local/share/private_ai/pages/`, so the agent
can find it again in later conversations. Text from the web is marked as
untrusted before the model sees it, but no marking makes a model immune to a
page written to mislead it. That is why file changes and commands go through
your approval rules (see Approvals), and why a wide rule deserves thought.

## The browser

When a page cannot be read plainly, the assistant opens it in Firefox, lets it
finish building itself, and reads what is there. This needs Firefox installed
and nothing else. It happens by itself in conversations and in research; the
model can also ask for it when a plain copy was clearly not the whole page.

    private-ai browser                     # open the assistant's browser as a window
    private-ai browser https://example.org/login
    private-ai browser --read URL          # print a page as the assistant would read it

* **A profile of its own.** The assistant's browser is separate from the one
  you browse with. It starts signed in to nothing, and your everyday history,
  passwords and sessions are out of its reach.
* **You sign in, not the assistant.** `private-ai browser` opens that profile
  as an ordinary window. Sign in there to the sites the assistant should be
  able to read, and close it. From then on a page on those sites is read as
  you would see it, so sign in only where you want that. Passwords are not
  saved in the profile; the sign-in itself (the site's cookies) is. The
  assistant cannot use the browser while the window is open.
* **It only looks.** The assistant can open an address and read the page. It
  cannot click, type or submit anything. The rule for addresses is the same as
  before: one that came from you or from a search or news result is read, and
  for any other you are asked.
* **A page runs.** Unlike plain reading, the site sees an ordinary browser:
  its scripts run, it loads what it likes from other sites, and this profile's
  cookies are sent. A page is closed as soon as it has been read.
* **Public sites only.** The browser is set up to refuse this machine and the
  local network, including when a page redirects there or tries to load
  something from there.
* **No window, and not left running.** It runs out of sight and is closed
  after two minutes without use.

Settings, if you need them, go under `[browser]` in the config file:
`enabled = false` turns it off, `firefox = "/path/to/firefox"` names the
program, `profile = "/path"` moves the profile, and `headless = false` shows
the window while the assistant reads. The profile is in
`~/.local/share/private_ai/browser/`, or, when Firefox is the snap package
(as on Ubuntu), in `~/snap/firefox/common/private_ai-browser/`, the only place
that package can use.

Some sites detect a remote-controlled browser and show a captcha instead of
the page. The assistant does not answer those. It tells you, and you can open
the page with `private-ai browser URL` and answer it yourself; the site may
then let later reads through, though some never do.

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
* **Pages that refuse are read with the browser**, if there is one, which is
  on this machine and uses the sign-ins you gave it. Those it cannot read
  either are skipped, and listed in the saved job. No third-party reading
  service is used.
* **A job that was cut off is marked interrupted**, whether the program was
  closed or the computer went off. If it had finished reading, the report can
  be written from the saved notes (`--retry`, or "Write the report again").
  Otherwise "Start again" begins a new job on the same topic.
* **Reports are kept** in `~/.local/share/private_ai/research/`, with the notes
  and sources behind them. The assistant sees the list of reports in every
  conversation and can read them, so you can discuss one later.

A report is only as good as the pages found and the model that read them.
Check the sources before relying on a claim.

## Files

The agent can read and write local files, but only with your approval. The
question says whether the contents would stay on your machine or go to a cloud
provider. Before a file is written you are shown what would be written: the
contents of a new file, or the lines that change in an existing one.

* The folder holding your API keys and approval rules is always off limits.
* The assistant cannot write to its own data folder with the file tools.
* Places that usually hold secrets (`~/.ssh`, `~/.aws`, browser profiles
  including the assistant's own,
  `.env` files, private keys and the like) are asked about every time, with a
  warning, whatever you have approved.
* Writing a file the system runs by itself (`~/.bashrc`, autostart entries,
  git hooks, system folders) also carries a warning every time.

These lists cover well-known locations and cannot be complete. Read the path
in every question.

## Coding

    cd your/project
    private-ai code                    # work on the code here, in the terminal
    private-ai code --resume           # carry on with the latest conversation
    private-ai code --read ~/code/mylib    # and let it read a library the project uses

In the terminal the project is the git repository around the current folder,
or the folder itself if there is none. In the web page, "New coding task"
asks for the project folder: it starts as the one `private-ai web` was started
in, and you can type any other, with folders you have used before offered as
you type. A task stays with the project it was started in.

A task can also be given folders it may read besides the project, such as a
library the code calls. Files in the project and in these folders are read
without a question, since naming a folder for the work is saying it may be
looked at. Nothing else changes: writing there still asks, `deny` and `ask`
rules still apply, and secrets still carry their warning. A search for files
with no folder given looks through all of them. Name them with `--read` (once for each
folder) or `/read FOLDER` in the terminal, or under "May also read" in the web
page, where they can be added and removed at any time. They are kept with the
conversation, not in the rules file.

A coding task has four more tools: edit part of a file, find files by name,
search file contents, and run a command. To test a program that asks questions, the assistant gives the answers along
with the command, and you see them in the approval question. An edit whose
text differs from the file only in spacing is applied to the place meant; the
change shown to you is the real one. A search gives up on a folder too
big to get through (200,000 files or 20 seconds) and says so, rather than
keeping you waiting. It can take up to 100 rounds of tool
use for one message. The conversation is saved after every round, so work that
is stopped, fails or is cut off keeps what was done up to then.

**A command is not confined.** Once it is allowed to run it can do anything
you can do: read any file, use the network, delete things. Approving commands
is the only control, so approve patterns with that in mind (`cat *` allows
reading any file and sending it to the model). Three things are done for you:
the API keys this program uses are removed from the command's environment,
nobody can type into it, and it is stopped after a time limit (two minutes
unless the model asks for more, ten at most) along with everything it started.
The status line shows the limit while a command runs. The model itself may
take as long as it needs, provided it is never silent for ten minutes.

## Approvals

Reading a file, changing a file and running a command each need approval. A
question can be answered five ways:

    y) yes, once
    s) yes, and the same for this session
    p) yes, always in this project
    g) yes, always, everywhere
    n) no

The wider answers cover more than the one request, and the question says what:
any file in a folder, or commands of a kind, such as `git status *`. A command
line joined with `&&`, `;` or `|` is allowed only when every part is. One that
writes its output to a file, runs in the background or uses another command's
output is asked about each time, and so are programs that run whatever they
are given, such as `sudo`, `bash -c` and `xargs`.

In a coding task a question about a command has one more answer: any command
in this task, without asking. It is for leaving a test-and-fix loop to run
while you are away, where every run is a different command line and no
pattern would cover them. It lasts until the program stops, is never saved,
and in the web page is a checkbox at the top of the task that you can also
set or clear yourself. `ask` and `deny` rules still apply, and so do the
warnings about secrets. It allows exactly what the paragraph on commands
above describes, so use it where that is acceptable.

An approval given while a local model is in use does not carry over to a cloud
provider: it is saved for local models only.

"Always" answers are saved in `~/.config/private_ai/approvals.toml`, which you
can also write by hand. The assistant cannot read or change that file.

    [[rule]]
    decision = "allow"            # allow, ask or deny
    action = "run"                # read, write or run
    match = "git status *"        # a command pattern, or a path pattern
    project = "/home/me/code/x"   # optional: only in this project
    local_only = true             # optional: only with a local model

    [[rule]]                      # never without asking, whatever else is allowed
    decision = "ask"
    action = "run"
    match = "git push *"

    [[rule]]                      # never at all
    decision = "deny"
    action = "write"
    match = "/etc/*"

When rules disagree the strictest wins: deny, then ask, then allow. For paths,
a folder covers everything in it, `*` matches anything including `/`, and a
pattern that is not absolute is taken inside the project.

    private-ai approvals               # the saved rules that apply in this project
    /rules                             # the same, in the terminal chat, with session approvals

The trust level sets the starting point, with `--trust` on `chat`, `code` and
`web`, or `trust = "..."` in the config file:

| Level | Allowed without asking |
|---|---|
| `ask` | Nothing, except that a coding task may read the folders it was given. This is the default. |
| `project` | Reading and changing files inside the project. Commands still ask. |
| `all` | Everything, including any command and the warnings above. |

`all` is for a machine where nothing matters, such as a throwaway virtual
machine. Deny rules still apply under it. On a machine that matters, stay with
`ask` and add `deny` and `ask` rules for what must never happen unseen.
