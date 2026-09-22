"""Command templates: the shell surface's equivalent of the DOM walk.

DOM affordances are enumerable, shell commands are not. Rather than let the
model write commands freely, the registry enumerates a curated set of
templates and binds their parameters from what is actually present -- the
files in the working directory, the task's own parameters. That produces
candidates mechanically, and Jev selects among them exactly as it selects
among page elements.

The point is coverage without generation. A task that only needs `grep`,
`cat` and `pytest` should never invoke the LLM at all; the Proposer exists
for what the registry cannot express, not as the default path.

Descriptions follow the same rule as DOM affordance descriptions: mechanical,
distinct, and stating what the command does rather than guessing why it might
be wanted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

# Files worth offering to read, and directories worth offering to search.
# Deliberately conservative: a registry that instantiates a template per file
# in a large tree would flood the candidate list and, per the duplicate-
# description finding, collapse choice confidence.
MAX_FILE_CANDIDATES = 12

INTERESTING_FILES = (
    "README.md",
    "pyproject.toml",
    "package.json",
    "Makefile",
    "Cargo.toml",
    "go.mod",
    "requirements.txt",
    "CLAUDE.md",
)

SKIP_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".mypy_cache",
}


@dataclass(frozen=True)
class CommandTemplate:
    """One command shape, with its parameters and its reversibility.

    ``reversible`` is the template author's declaration, not a model
    judgement. It does not replace the irreversibility gate -- the guard panel
    still runs -- but it lets the registry avoid offering a destructive
    command as a routine candidate in the first place.
    """

    name: str
    argv: tuple[str, ...]
    description: str
    reversible: bool = True
    # Produces zero or more concrete (suffix, argv, description) bindings from
    # the observed working directory. A template with no binder yields itself.
    binder: Callable[[Path], list[tuple[str, tuple[str, ...], str]]] | None = None

    def instantiate(self, root: Path) -> list[tuple[str, tuple[str, ...], str]]:
        if self.binder is None:
            return [(self.name, self.argv, self.description)]
        return self.binder(root)


def _visible_files(root: Path, limit: int = MAX_FILE_CANDIDATES) -> list[Path]:
    """Files worth offering, nearest the root first."""
    found: list[Path] = []
    for name in INTERESTING_FILES:
        candidate = root / name
        if candidate.is_file():
            found.append(candidate)

    for path in sorted(root.rglob("*")):
        if len(found) >= limit:
            break
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path in found:
            continue
        if path.suffix in (".py", ".ts", ".js", ".go", ".rs", ".md", ".toml", ".json"):
            found.append(path)

    return found[:limit]


def _source_directories(root: Path) -> list[Path]:
    candidates = [root / name for name in ("src", "lib", "app", "tests", "test")]
    return [d for d in candidates if d.is_dir()]


def _bind_cat(root: Path) -> list[tuple[str, tuple[str, ...], str]]:
    bindings = []
    for path in _visible_files(root):
        relative = path.relative_to(root).as_posix()
        bindings.append(
            (
                f"cat__{relative.replace('/', '_')}",
                ("cat", relative),
                f"Run shell command: cat {relative} (read the contents of {relative})",
            )
        )
    return bindings


def _bind_grep(root: Path) -> list[tuple[str, tuple[str, ...], str]]:
    bindings = []
    for directory in _source_directories(root) or [root]:
        relative = directory.relative_to(root).as_posix() if directory != root else "."
        bindings.append(
            (
                f"grep__{relative.replace('/', '_')}",
                ("grep", "-rn", "{pattern}", relative),
                f"Run shell command: grep -rn <pattern> {relative} "
                f"(search {relative} for a text pattern)",
            )
        )
    return bindings


def _bind_ls(root: Path) -> list[tuple[str, tuple[str, ...], str]]:
    bindings = [("ls__root", ("ls", "-la"), "Run shell command: ls -la (list the working directory)")]
    for directory in _source_directories(root):
        relative = directory.relative_to(root).as_posix()
        bindings.append(
            (
                f"ls__{relative.replace('/', '_')}",
                ("ls", "-la", relative),
                f"Run shell command: ls -la {relative} (list the contents of {relative})",
            )
        )
    return bindings


REGISTRY: tuple[CommandTemplate, ...] = (
    CommandTemplate(
        name="ls",
        argv=("ls", "-la"),
        description="Run shell command: ls -la (list the working directory)",
        binder=_bind_ls,
    ),
    CommandTemplate(
        name="cat",
        argv=("cat", "{file}"),
        description="Run shell command: cat <file>",
        binder=_bind_cat,
    ),
    CommandTemplate(
        name="grep",
        argv=("grep", "-rn", "{pattern}", "{path}"),
        description="Run shell command: grep -rn <pattern> <path>",
        binder=_bind_grep,
    ),
    CommandTemplate(
        name="git_status",
        argv=("git", "status", "--short"),
        description="Run shell command: git status --short (show the working tree state)",
    ),
    CommandTemplate(
        name="git_diff",
        argv=("git", "diff"),
        description="Run shell command: git diff (show uncommitted changes)",
    ),
    CommandTemplate(
        name="git_log",
        argv=("git", "log", "--oneline", "-20"),
        description="Run shell command: git log --oneline -20 (show recent commits)",
    ),
    CommandTemplate(
        name="pytest",
        argv=("uv", "run", "pytest", "-q"),
        description="Run shell command: uv run pytest -q (run the test suite)",
    ),
    CommandTemplate(
        name="uv_sync",
        argv=("uv", "sync"),
        description="Run shell command: uv sync (install project dependencies)",
        reversible=False,
    ),
    CommandTemplate(
        name="find_name",
        argv=("find", ".", "-name", "{pattern}", "-not", "-path", "*/.git/*"),
        description="Run shell command: find . -name <pattern> (locate files by name)",
    ),
    CommandTemplate(
        name="wc_lines",
        argv=("wc", "-l", "{file}"),
        description="Run shell command: wc -l <file> (count lines in a file)",
    ),
)


@dataclass(frozen=True)
class ShellCandidate:
    key: str
    description: str
    argv: tuple[str, ...]
    reversible: bool
    # Parameters still needing a value, e.g. a grep pattern. The controller
    # binds these from task parameters, or asks Jev to select among them.
    unbound: tuple[str, ...] = ()

    @property
    def is_ready(self) -> bool:
        return not self.unbound


def _unbound_parameters(argv: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        part[1:-1] for part in argv if part.startswith("{") and part.endswith("}")
    )


def enumerate_candidates(
    root: Path,
    params: dict[str, str] | None = None,
    include_unready: bool = True,
) -> dict[str, ShellCandidate]:
    """Produce shell candidates from what is actually present.

    ``params`` binds free parameters (a grep pattern, a filename) from the
    task's own parameters. A template whose parameters cannot be bound is
    still offered when ``include_unready`` is set, because Jev selecting it is
    what tells the controller a value is needed -- which is exactly when the
    Value generator is invoked.
    """
    params = params or {}
    candidates: dict[str, ShellCandidate] = {}

    for template in REGISTRY:
        for key, argv, description in template.instantiate(root):
            bound = tuple(
                params.get(part[1:-1], part)
                if part.startswith("{") and part.endswith("}")
                else part
                for part in argv
            )
            unbound = _unbound_parameters(bound)
            if unbound and not include_unready:
                continue
            candidates[f"sh__{key}"] = ShellCandidate(
                key=f"sh__{key}",
                description=description,
                argv=bound,
                reversible=template.reversible,
                unbound=unbound,
            )

    return candidates


def describe_state(root: Path, last_result: str | None = None) -> str:
    """Render the SHELL section of the Jev state string."""
    entries = []
    for path in sorted(root.iterdir())[:20]:
        if path.name in SKIP_DIRECTORIES:
            continue
        entries.append(f"{path.name}/" if path.is_dir() else path.name)

    lines = [f"cwd={root}", f"contents: {', '.join(entries) if entries else '(empty)'}"]
    if last_result:
        lines.append("")
        lines.append(last_result)
    else:
        lines.append("No commands have been run yet.")
    return "\n".join(lines)
