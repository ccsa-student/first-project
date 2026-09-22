"""Sandboxed shell execution.

The irreversibility panel classifies shell commands correctly (`rm -rf ./build`
0.66 against `ls -la` 0.16), but that is a gate, not a containment boundary.
A classifier sitting at 0.66 is a judgement under uncertainty; containment
must not be. So everything here holds regardless of what Jev returns:

    jail          paths resolving outside the project root are refused
    argv          commands execute as vectors, never through a shell
    deny-list     destructive and privilege patterns are refused outright
    timeout       every command has a wall-clock ceiling
    output cap    stdout and stderr are truncated before entering the state
    env scrub     the subprocess environment is an allow-list; no secrets

Command output is untrusted input on the same footing as page text. It can
carry fetched web content, attacker-controlled file contents, or hostile
filenames, and it flows into the Jev state string, so the injection gate
covers it too.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Per-command wall clock. Long-running work needs an explicit
# background-and-poll template rather than a bigger number here.
DEFAULT_TIMEOUT = 30.0

# Output beyond this is truncated before it reaches the state string. A large
# command output is as capable of blowing Jev's 32,768-token ceiling as a
# large DOM.
MAX_OUTPUT_CHARS = 8_000

# Shell control operators. A registry template executes as an argv vector so
# these cannot appear, but a free-form LLM-proposed command is parsed and
# refused if it contains any -- rather than being handed to a shell.
CONTROL_OPERATORS = (";", "&&", "||", "|", ">", ">>", "<", "`", "$(", "\n")

# Refused regardless of the irreversibility score. This list is about
# containment, not about judgement: everything here is refused even when Jev
# rates it harmless, and the guard panel still runs on everything that is not.
DENIED_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^sudo\b", "privilege escalation"),
    (r"^su\b", "privilege escalation"),
    (r"^doas\b", "privilege escalation"),
    (r"\brm\s+(-[a-zA-Z]*\s+)*/(\s|$)", "recursive delete of the filesystem root"),
    (r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*\s+)*(~|\$HOME)", "recursive delete of the home directory"),
    (r"\bmkfs\b", "filesystem formatting"),
    (r"\bdd\b.*\bof=/dev/", "raw device write"),
    (r":\(\)\s*\{.*\};:", "fork bomb"),
    (r"\bchmod\s+(-[a-zA-Z]*\s+)*777\s+/", "permissions change on the filesystem root"),
    (r"\bcurl\b.*\|\s*(ba)?sh", "piping a download into a shell"),
    (r"\bwget\b.*\|\s*(ba)?sh", "piping a download into a shell"),
    (r"\bgit\s+push\b.*--force", "force push rewrites published history"),
    (r"\bgit\s+reset\b.*--hard", "discards uncommitted work irrecoverably"),
    (r"\bgit\s+clean\b.*-[a-zA-Z]*f", "deletes untracked files irrecoverably"),
    (r"\bnpm\s+publish\b", "publishes a package"),
    (r"\b(twine|uv)\s+publish\b", "publishes a package"),
    (r"\bshutdown\b|\breboot\b|\bhalt\b", "host lifecycle"),
    (r"\bhistory\s+-c\b", "clears shell history"),
    (r"/etc/(passwd|shadow|sudoers)", "credential file access"),
    # The negative lookahead has to span the rest of the filename: a plain
    # `(?!\.pub)` is satisfied by backtracking the `+` and so matches
    # `id_rsa.pub` anyway.
    (r"\.ssh/id_[a-z0-9_]+(?![\w.]*\.pub)", "private key access"),
    (r"\.aws/credentials", "credential file access"),
)

# Deliberately NOT denied: `env` and `printenv`.
#
# An earlier draft refused them as "would surface the environment, including
# secrets". That is defence at the wrong layer. The subprocess environment is
# already built from an allow-list, so there is nothing secret in it to
# surface, and denying the commands only blocked legitimate debugging while
# implying a protection that the allow-list is actually providing. If the
# allow-list ever regressed, denying `env` would hide the regression rather
# than prevent it.

# The subprocess environment is built from this allow-list only. Anything not
# named here -- notably OPENROUTER_API_KEY and TYPESAFE_API_KEY -- is absent
# from every command the agent runs.
ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TERM",
    "TZ",
    "PWD",
    "SHELL",
    "USER",
)


class ShellRefused(Exception):
    """The sandbox refused to run a command. Never a failure of the command."""

    def __init__(self, command: str, reason: str) -> None:
        self.command = command
        self.reason = reason
        super().__init__(f"refused: {reason} ({command!r})")


@dataclass(frozen=True)
class ShellResult:
    argv: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    truncated: bool
    timed_out: bool
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary(self) -> str:
        """Render for the SHELL section of the Jev state string.

        The exit code and the tail of stderr are never truncated away: they
        are what the is_error gate keys on for this surface.
        """
        parts = [f"$ {' '.join(self.argv)}", f"Exit code: {self.exit_code}"]
        if self.timed_out:
            parts.append(f"TIMED OUT after {self.duration_s:.1f}s")
        if self.stdout.strip():
            parts.append(f"stdout:\n{self.stdout}")
        if self.stderr.strip():
            parts.append(f"stderr:\n{self.stderr}")
        if self.truncated:
            parts.append("(output truncated)")
        return "\n".join(parts)


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """Keep the head and the tail; the middle is where the least signal is.

    A failing command's message is usually at the end, and a listing's shape
    is usually at the start, so dropping the middle preserves both.
    """
    if len(text) <= limit:
        return text, False
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n... [{len(text) - limit} characters omitted] ...\n{tail}", True


class Sandbox:
    """Runs commands under containment guarantees."""

    def __init__(
        self,
        root: Path,
        timeout: float = DEFAULT_TIMEOUT,
        max_output: int = MAX_OUTPUT_CHARS,
    ) -> None:
        self.root = root.resolve()
        if not self.root.is_dir():
            raise ValueError(f"sandbox root is not a directory: {self.root}")
        self.timeout = timeout
        self.max_output = max_output

    # -- containment ------------------------------------------------------

    def check_denied(self, command: str) -> str | None:
        """Return the reason this command is refused, or None."""
        for pattern, reason in DENIED_PATTERNS:
            if re.search(pattern, command):
                return reason
        return None

    def resolve_in_jail(self, candidate: str | Path) -> Path:
        """Resolve a path and require it to stay inside the root.

        Resolution happens before the check, so `../../etc/passwd` and a
        symlink pointing outside are both caught. ``strict=False`` lets a
        not-yet-existing path be validated, which matters for output files.
        """
        path = Path(candidate)
        absolute = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        if absolute != self.root and self.root not in absolute.parents:
            raise ShellRefused(str(candidate), f"path escapes the sandbox root {self.root}")
        return absolute

    def parse_freeform(self, command: str) -> tuple[str, ...]:
        """Parse an LLM-proposed command string into an argv vector.

        Refuses rather than shell-interpreting anything containing a control
        operator. The agent does not get a shell; it gets a process launcher.
        """
        for operator in CONTROL_OPERATORS:
            if operator in command:
                raise ShellRefused(
                    command,
                    f"contains the shell control operator {operator!r}; "
                    "commands run as argument vectors, not through a shell",
                )

        reason = self.check_denied(command)
        if reason:
            raise ShellRefused(command, reason)

        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise ShellRefused(command, f"could not be parsed: {exc}") from exc

        if not argv:
            raise ShellRefused(command, "empty command")
        return tuple(argv)

    def _environment(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k in ENV_ALLOWLIST}
        env["PWD"] = str(self.root)
        return env

    # -- execution --------------------------------------------------------

    def run(self, argv: tuple[str, ...] | list[str], cwd: Path | None = None) -> ShellResult:
        """Execute an argv vector inside the jail.

        Every path-shaped argument is jail-checked first. Arguments that are
        not paths simply do not resolve to anything outside the root, so the
        check is harmless for them.
        """
        argv = tuple(argv)
        if not argv:
            raise ShellRefused("", "empty command")

        joined = " ".join(argv)
        reason = self.check_denied(joined)
        if reason:
            raise ShellRefused(joined, reason)

        for argument in argv[1:]:
            # Only check things that look like paths. A bare flag or a grep
            # pattern is not a path, and forcing it through resolution would
            # produce confusing refusals.
            if argument.startswith("-"):
                continue
            if "/" in argument or argument.startswith(".."):
                self.resolve_in_jail(argument)

        working_directory = self.resolve_in_jail(cwd) if cwd else self.root

        import time

        started = time.perf_counter()
        timed_out = False
        try:
            completed = subprocess.run(
                argv,
                cwd=working_directory,
                env=self._environment(),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                # Never shell=True. The whole containment story depends on it.
                shell=False,
            )
            exit_code = completed.returncode
            stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            stderr = f"{stderr}\nCommand exceeded the {self.timeout:.0f}s limit."
        except FileNotFoundError:
            exit_code = 127
            stdout, stderr = "", f"{argv[0]}: command not found"
        except PermissionError as exc:
            exit_code = 126
            stdout, stderr = "", str(exc)

        duration = time.perf_counter() - started
        stdout, out_truncated = _truncate(stdout, self.max_output)
        stderr, err_truncated = _truncate(stderr, self.max_output)

        return ShellResult(
            argv=argv,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            truncated=out_truncated or err_truncated,
            timed_out=timed_out,
            duration_s=duration,
        )

    def run_freeform(self, command: str) -> ShellResult:
        """Parse and run an LLM-proposed command. Refuses before executing."""
        return self.run(self.parse_freeform(command))
