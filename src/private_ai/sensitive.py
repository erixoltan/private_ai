"""Which paths deserve a warning before the assistant touches them.

Two kinds:
* Secrets: keys, passwords and logins. Reading one could send it to a cloud
  provider; replacing one could lock the user out or let someone else in.
* Files the system runs: shell startup files, autostart entries, git hooks,
  system folders. Writing one is a way to run code later without asking.

The lists are of well-known locations and names. They cannot be complete, so a
path that is not flagged is not thereby safe: it still needs the usual approval.
"""

from __future__ import annotations

import functools
import os
import re
from fnmatch import translate
from pathlib import Path

# Relative to the home folder. A folder covers everything inside it.
_SECRET_HOME = (
    ".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".password-store",
    ".config/gcloud", ".config/gh", ".config/op", ".local/share/keyrings",
    ".mozilla", ".thunderbird", ".config/google-chrome", ".config/chromium",
    ".config/BraveSoftware", ".config/microsoft-edge",
    ".netrc", ".npmrc", ".pypirc", ".git-credentials", ".pgpass", ".my.cnf",
)
_SECRET_SYSTEM = ("/etc/shadow", "/etc/gshadow", "/etc/sudoers", "/etc/sudoers.d", "/etc/ssh", "/root")
_SECRET_NAMES = (
    ".env", ".env.*", "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*",
    "*.pem", "*.key", "*.p12", "*.pfx", "*.kdbx", "*.keystore", "credentials.json",
)
_NOT_SECRET_NAMES = (".env.example", ".env.sample", ".env.template", "*.pub")

_RUN_HOME = (
    ".bashrc", ".bash_profile", ".bash_login", ".bash_logout", ".bash_aliases", ".profile",
    ".zshrc", ".zshenv", ".zprofile", ".zlogin", ".zlogout", ".xinitrc", ".xprofile", ".xsessionrc",
    ".pam_environment", ".gitconfig", ".config/git", ".config/fish", ".config/autostart",
    ".config/systemd", ".config/environment.d", ".local/bin", ".local/share/applications",
)
_RUN_SYSTEM = ("/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/boot", "/opt", "/var/spool/cron")


def _within(path: Path, roots: list[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _home(names: tuple[str, ...]) -> list[Path]:
    return _under(os.path.expanduser("~"), names)


@functools.lru_cache(maxsize=8)
def _under(home: str, names: tuple[str, ...]) -> list[Path]:
    # Worked out once: a file search asks about every file it passes.
    resolved = Path(home).resolve()
    return [resolved / name for name in names]


_SYSTEM = {names: [Path(p) for p in names] for names in (_SECRET_SYSTEM, _RUN_SYSTEM)}


_NAMES = re.compile("|".join(translate(p) for p in _SECRET_NAMES))
_NOT_NAMES = re.compile("|".join(translate(p) for p in _NOT_SECRET_NAMES))


def secret_name(name: str) -> bool:
    """Whether a file or folder of this name usually holds keys or passwords, wherever it is."""
    return bool(_NAMES.match(name)) and not _NOT_NAMES.match(name)


def secret_places_in(folder: Path) -> set[str]:
    """The names directly inside a folder that are places for secrets. With
    secret_name, this lets a walk down from a folder skip what secret() would
    flag, without asking about every file. Give a resolved path."""
    return {place.name for place in (*_home(_SECRET_HOME), *_SYSTEM[_SECRET_SYSTEM]) if place.parent == folder}


def secret(path: Path) -> str | None:
    """Why this path probably holds secrets, or None. Give a resolved path."""
    if _within(path, _home(_SECRET_HOME)) or _within(path, _SYSTEM[_SECRET_SYSTEM]):
        return "This location usually holds keys, passwords or logins."
    if secret_name(path.name):
        return "Files with this name usually hold keys or passwords."
    return None


def runs_code(path: Path) -> str | None:
    """Why writing this path could run code later, or None. Give a resolved path."""
    if _within(path, _home(_RUN_HOME)):
        return "This is run or read automatically, for example when you log in or open a terminal."
    parts = path.parts
    if ".git" in parts:
        inside = parts[parts.index(".git") + 1:]
        if inside[:1] == ("hooks",) or inside == ("config",):
            return "Git runs what is in here during ordinary git commands."
    if _within(path, _SYSTEM[_RUN_SYSTEM]):
        return "This is a system location."
    return None


def mentioned(command: str, also: tuple[Path, ...] = ()) -> str | None:
    """Why a command deserves a second look: it names a place that holds
    secrets, or one of the folders in `also`. Only what is written out in the
    command can be seen, so this catches the plain cases and no others."""
    for word in re.split(r"[\s'\"=;|&<>()]+", command):
        if not word:
            continue
        path = Path(os.path.expanduser(word.replace("${HOME}", "~").replace("$HOME", "~")))
        if _within(path, list(also)):
            return f"This command names a folder that is otherwise off limits: {word}"
        if secret(path):
            return f"This command names a place that usually holds keys or passwords: {word}"
    return None
