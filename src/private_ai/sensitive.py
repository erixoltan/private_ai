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

from fnmatch import fnmatch
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
    home = Path.home().resolve()
    return [home / name for name in names]


def secret(path: Path) -> str | None:
    """Why this path probably holds secrets, or None. Give a resolved path."""
    if _within(path, _home(_SECRET_HOME)) or _within(path, [Path(p) for p in _SECRET_SYSTEM]):
        return "This location usually holds keys, passwords or logins."
    name = path.name
    if any(fnmatch(name, p) for p in _SECRET_NAMES) and not any(fnmatch(name, p) for p in _NOT_SECRET_NAMES):
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
    if _within(path, [Path(p) for p in _RUN_SYSTEM]):
        return "This is a system location."
    return None
