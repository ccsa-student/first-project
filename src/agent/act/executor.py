"""Executes a chosen action on the appropriate surface.

The executor is deliberately thin and deliberately last in the chain. By the
time an action reaches it, it has won a Jev choice, cleared the gate panel and
cleared the guard panel, and -- if irreversible -- been confirmed by a human.
Nothing here re-litigates those decisions.

What it does own is the refusal boundary: the sandbox can still refuse a
command the classifier was happy with, and that refusal is an outcome the loop
records and continues from, not an exception that kills the run.
"""

from __future__ import annotations

from pathlib import Path

from ..tools.shell import Sandbox, ShellRefused
from .actions import ActionOutcome, Candidate


class BrowserNotAvailable(RuntimeError):
    """Raised when a browser action is attempted before the DOM layer exists."""


class Executor:
    def __init__(self, sandbox: Sandbox | None = None, root: Path | None = None) -> None:
        if sandbox is None and root is None:
            raise ValueError("Executor needs a sandbox or a root directory")
        self.sandbox = sandbox or Sandbox(root=root)  # type: ignore[arg-type]

    def execute(self, candidate: Candidate) -> ActionOutcome:
        if not candidate.is_ready:
            return ActionOutcome(
                candidate=candidate,
                ok=False,
                observation=(
                    f"Cannot run: {', '.join(candidate.unbound)} still needs a value."
                ),
                refused_reason=f"unbound parameter(s): {', '.join(candidate.unbound)}",
            )

        if candidate.surface == "shell":
            return self._execute_shell(candidate)

        raise BrowserNotAvailable(
            "browser actions require the observation layer, which is not built yet"
        )

    def _execute_shell(self, candidate: Candidate) -> ActionOutcome:
        assert candidate.argv is not None
        try:
            result = self.sandbox.run(candidate.argv)
        except ShellRefused as exc:
            # A refusal is a normal outcome: containment held. The loop
            # records it and carries on, so the agent can try something else
            # rather than dying.
            return ActionOutcome(
                candidate=candidate,
                ok=False,
                observation=f"The sandbox refused this command: {exc.reason}",
                refused_reason=exc.reason,
            )

        return ActionOutcome(
            candidate=candidate,
            ok=result.ok,
            observation=result.summary(),
            exit_code=result.exit_code,
        )
