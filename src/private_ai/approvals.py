"""Who decides whether the assistant may do something: the user's rules, and
failing those, the user.

Five things are governed: reading a file ("read"), changing one ("write"),
running a command ("run"), and in the browser clicking something on a web
site ("click") and typing into one ("type"). For each request the answer is
allow, ask or deny.

Rules come from three places:

* The session: approvals given with "for this session". Kept in memory only.
* The rules file, approvals.toml, next to the config file. It holds approvals
  given with "always", for one project or for everywhere, and anything the
  user writes there by hand. It is outside every project and inside the folder
  the assistant can never touch, so neither the assistant nor a downloaded
  project can grant itself anything.
* The trust level, a starting point chosen when the program starts:
    ask      nothing is allowed without a rule (the default)
    project  files inside the project may be read and changed without asking
    all      everything is allowed, commands included; for a machine where
             nothing matters
* The folders of a coding task: the project it was started in, and any the
  user named as ones it may also read, such as a library the project uses.
  Reading in them is allowed, since naming a folder for the work is saying
  it may be looked at. Changing anything in them is asked like any other change.

When rules disagree the strictest wins: deny, then ask, then allow. With no
rule, the user is asked.

A rule in the file looks like this:

    [[rule]]
    decision = "allow"            # allow, ask or deny
    action = "run"                # read, write, run, click or type
    match = "git status *"        # a command pattern, a path pattern, or a site
    project = "/home/me/code/x"   # optional: only when working in this project
    local_only = true             # optional: only while a local model is in use

Path patterns: a folder covers everything under it; * matches anything,
including "/"; a pattern that is not absolute is taken inside the project.

Command patterns: "git status *" covers "git status" with any arguments, and
"*" covers every command. A command line joined with ;, && , || or | is allowed
only if every part is. One that redirects output, runs in the background or
substitutes another command's output is too much to judge by pattern, so only
"*" allows it.

Sites: "example.org" covers that site and every name under it, such as
www.example.org, and "*" covers every site. A rule for a site says nothing
about what is clicked or typed there, so allow one only for sites where
nothing much can go wrong, and see tools.py for what is asked regardless.

Approvals given under a local model are saved as local_only: nothing was said
about sending that file or that command's output to a cloud provider.
"""

from __future__ import annotations

import os
import re
import shlex
import tomllib
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from private_ai.providers import config_path

ACTIONS = ("read", "write", "run", "click", "type")
SITE_ACTIONS = ("click", "type")
DECISIONS = ("allow", "ask", "deny")
TRUST_LEVELS = ("ask", "project", "all")

_JOINERS = {";", "&&", "||", "|"}
# Programs whose own subcommand is what matters, so a pattern keeps two words.
_TWO_WORDS = {
    "git", "npm", "pnpm", "yarn", "cargo", "docker", "kubectl", "uv", "pip", "pip3", "go", "apt", "apt-get",
    "systemctl", "gh", "make", "dotnet", "brew",
}
# Programs that run whatever they are given: a pattern for one would allow anything.
_WRAPPERS = {
    "sudo", "su", "doas", "env", "xargs", "bash", "sh", "zsh", "fish", "eval", "exec", "nohup", "time",
    "command", "watch", "timeout", "nice", "ssh",
}


_HARMLESS = re.compile(r"(?<![\w>&])(2>&1|[12]?>\s*/dev/null|&>\s*/dev/null)(?![\w/.])")


def default_trust() -> str:
    """The trust level set in the config file, or ask."""
    path = config_path()
    if path.exists():
        with path.open("rb") as f:
            return str(tomllib.load(f).get("trust", "ask"))
    return "ask"


def rules_path() -> Path:
    return config_path().parent / "approvals.toml"


def find_project(start: Path | None = None) -> Path:
    """The project a folder belongs to: the nearest folder above it that is a
    git repository, or else the folder itself."""
    start = (start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        if (folder / ".git").exists():
            return folder
    return start


def chosen_folder(text: str, base: Path | None = None) -> Path:
    """The folder the user named, for a coding task to work in or to read.
    Raises ValueError, saying why, if it cannot be one."""
    text = text.strip()
    if not text:
        raise ValueError("no folder was given.")
    given = Path(text).expanduser()
    if not given.is_absolute():
        if base is None:
            raise ValueError(f"give the whole path, starting with / or ~ (got: {text})")
        given = base / given
    folder = given.resolve()
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a folder.")
    protected = config_path().resolve().parent
    if folder == protected or protected in folder.parents:
        raise ValueError("that folder holds API keys and is off limits.")
    return folder


@dataclass(frozen=True)
class Rule:
    decision: str
    action: str
    match: str
    project: str | None = None
    local_only: bool = False
    source: str = "file"  # file, session, trust or chosen

    def describe(self) -> str:
        where = f" in {self.project}" if self.project else ""
        local = " (local models only)" if self.local_only else ""
        return f"{self.decision} {self.action} {self.match}{where}{local}"


def split_command(command: str) -> list[str] | None:
    """The simple commands a command line is made of, or None if it does
    anything a pattern cannot judge (redirection, substitution, background)."""
    command = _HARMLESS.sub(" ", command)  # joining or discarding output changes nothing a rule cares about
    if any(mark in command for mark in ("`", "$(", "\n", "<(", ">(")):
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    parts, current = [], []
    for token in tokens:
        if token in _JOINERS:
            if not current:
                return None
            parts.append(" ".join(current))
            current = []
        elif token and all(c in "();<>|&" for c in token):
            return None
        else:
            current.append(token)
    if not current:
        return None
    return [*parts, " ".join(current)]


def command_pattern(part: str) -> str | None:
    """The pattern offered for approving commands like this one more widely."""
    words = part.split()
    first = words[0]
    if "=" in first or first in _WRAPPERS:
        return None  # a variable assignment, or a program that runs anything
    if first in ("python", "python3") and len(words) > 2 and words[1] == "-m":
        return f"{first} -m {words[2]} *"
    if first in _TWO_WORDS and len(words) > 1 and not words[1].startswith("-"):
        return f"{first} {words[1]} *"
    return f"{first} *"


def _site_matches(pattern: str, host: str) -> bool:
    pattern, host = pattern.strip().lower().lstrip("."), host.lower()
    return pattern == "*" or fnmatchcase(host, pattern) or fnmatchcase(host, "*." + pattern)


def _command_matches(pattern: str, part: str) -> bool:
    if pattern == "*":
        return True
    if pattern.endswith(" *"):
        base = pattern[:-2]
        return part == base or part.startswith(base + " ")
    return fnmatchcase(part, pattern)


class Policy:
    def __init__(self, project: Path, local: bool, trust: str = "ask", path: Path | None = None):
        if trust not in TRUST_LEVELS:
            raise ValueError(f"trust must be one of: {', '.join(TRUST_LEVELS)}")
        self.project = project.resolve()
        self.local = local
        self.trust = trust
        self.path = path or rules_path()
        self.session: list[Rule] = []
        self.folders: list[Path] = []  # folders the user named as readable, besides the project
        self.project_readable = False  # a coding task may read its own project
        self.problem = ""  # why the rules file could not be used, if it could not
        self._loaded: tuple[float, list[Rule]] | None = None

    # ---- the rules in force ----

    def _file_rules(self) -> list[Rule]:
        try:
            changed = self.path.stat().st_mtime
        except OSError:
            self.problem = ""
            return []
        if self._loaded and self._loaded[0] == changed:
            return self._loaded[1]
        rules = []
        try:
            with self.path.open("rb") as f:
                entries = tomllib.load(f).get("rule", [])
            for entry in entries:
                rule = Rule(
                    str(entry["decision"]), str(entry["action"]), str(entry["match"]),
                    str(entry["project"]) if entry.get("project") else None, bool(entry.get("local_only", False)),
                )
                if rule.decision not in DECISIONS or rule.action not in ACTIONS or not rule.match:
                    raise ValueError(f"bad rule: {rule.describe()}")
                rules.append(rule)
            self.problem = ""
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            # A file that cannot be read grants nothing, and that is said rather than hidden.
            rules, self.problem = [], f"{self.path} could not be used, so everything is asked: {e}"
        self._loaded = (changed, rules)
        return rules

    def _trust_rules(self) -> list[Rule]:
        if self.trust == "project":
            folder = str(self.project)
            return [Rule("allow", "read", folder, source="trust"), Rule("allow", "write", folder, source="trust")]
        if self.trust == "all":
            return [Rule("allow", action, "*", source="trust") for action in ACTIONS]
        return []

    def rules(self) -> list[Rule]:
        """Every rule that applies here and now."""
        return [
            rule
            for rule in [
                *self._trust_rules(),
                *(Rule("allow", "read", str(folder), source="chosen") for folder in self.readable()),
                *self.session,
                *self._file_rules(),
            ]
            if (rule.project is None or Path(rule.project).expanduser() == self.project)
            and (self.local or not rule.local_only)
        ]

    def readable(self) -> list[Path]:
        """The folders that may be read because the user named them for this work."""
        return [*([self.project] if self.project_readable else []), *self.folders]

    # ---- deciding ----

    def _path_matches(self, pattern: str, path: str) -> bool:
        if pattern == "*":
            return True
        pattern = os.path.expanduser(pattern)
        if not os.path.isabs(pattern):
            pattern = str(self.project / pattern)
        pattern = pattern.rstrip("/")
        return fnmatchcase(path, pattern) or fnmatchcase(path, pattern + "/*")

    def decide(self, action: str, target: str) -> tuple[str, Rule | None]:
        """allow, ask or deny, and the rule that settled it (None when no rule did)."""
        rules = [rule for rule in self.rules() if rule.action == action]
        if action != "run":
            matches = _site_matches if action in SITE_ACTIONS else self._path_matches
            for decision in ("deny", "ask", "allow"):
                for rule in rules:
                    if rule.decision == decision and matches(rule.match, target):
                        return decision, rule
            return "ask", None

        parts = split_command(target)
        pieces = [target.strip(), *(parts or [])]
        for decision in ("deny", "ask"):
            for rule in rules:
                if rule.decision == decision and any(_command_matches(rule.match, piece) for piece in pieces):
                    return decision, rule
        allows = [rule for rule in rules if rule.decision == "allow"]
        if parts is None:  # too much for a pattern: only a blanket rule allows it
            blanket = next((rule for rule in allows if rule.match == "*"), None)
            return ("allow", blanket) if blanket else ("ask", None)
        used = None
        for part in parts:
            used = next((rule for rule in allows if _command_matches(rule.match, part)), None)
            if used is None:
                return "ask", None
        return "allow", used

    def wider(self, action: str, target: str) -> list[str]:
        """Patterns that, if approved, would cover this request and others like
        it. Empty when the request is one that should only ever be approved once."""
        if action in SITE_ACTIONS:
            return [target]
        if action != "run":
            path = Path(target)
            inside = path == self.project or self.project in path.parents
            return [str(self.project if inside else path if path.is_dir() else path.parent)]
        parts = split_command(target)
        if parts is None:
            return []
        allows = [rule for rule in self.rules() if rule.action == "run" and rule.decision == "allow"]
        patterns = []
        for part in parts:
            if any(_command_matches(rule.match, part) for rule in allows):
                continue
            pattern = command_pattern(part)
            if pattern is None:
                return []
            if pattern not in patterns:
                patterns.append(pattern)
        return patterns

    # ---- granting ----

    @property
    def free_run(self) -> bool:
        """Whether any command may run without asking, for this session."""
        return any(r.action == "run" and r.decision == "allow" and r.match == "*" for r in self.session)

    def set_free_run(self, on: bool) -> None:
        self.session = [r for r in self.session if not (r.action == "run" and r.match == "*")]
        if on:
            self.grant("run", "*", "session")

    def grant(self, action: str, match: str, scope: str) -> Rule:
        """Allow an action from now on. scope: session, project or everywhere."""
        rule = Rule(
            "allow", action, match,
            project=str(self.project) if scope == "project" else None,
            local_only=self.local,
            source="session" if scope == "session" else "file",
        )
        if scope == "session":
            self.session.append(rule)
        else:
            self._append(rule)
        return rule

    def _append(self, rule: Rule) -> None:
        """Add a rule to the end of the file, leaving what the user wrote there untouched."""
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        quote = lambda text: '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'  # noqa: E731
        lines = ["", "[[rule]]", 'decision = "allow"', f"action = {quote(rule.action)}", f"match = {quote(rule.match)}"]
        if rule.project:
            lines.append(f"project = {quote(rule.project)}")
        if rule.local_only:
            lines.append("local_only = true")
        new = not self.path.exists()
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            if new:
                f.write("# What the assistant may do without asking. See the README for the format.\n")
            f.write("\n".join(lines) + "\n")
        self._loaded = None
