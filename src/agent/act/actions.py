"""Action types shared by both surfaces.

A ``Candidate`` is something Jev may choose. It carries its provenance --
whether it came from the DOM walk, the command registry, or the LLM Proposer
-- because that distinction drives three things: how the dashboard renders it,
what the metrics count, and the fact that an LLM-originated action is never
allowed to reach the executor without having won a Jev choice first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Surface = Literal["browser", "shell"]
Provenance = Literal["dom", "registry", "llm"]


@dataclass(frozen=True)
class Candidate:
    """One selectable action.

    ``key`` is what Jev returns; ``description`` is what it sees. The two are
    kept apart so a description can be reworded without invalidating labels
    that reference the key.
    """

    key: str
    description: str
    surface: Surface
    provenance: Provenance
    # Shell actions carry an already-bound argument vector. Browser actions
    # carry an element reference, added when the DOM layer lands.
    argv: tuple[str, ...] | None = None
    element_ref: str | None = None
    reversible: bool = True
    unbound: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_ready(self) -> bool:
        return not self.unbound

    def __post_init__(self) -> None:
        if self.surface == "shell" and self.argv is None:
            raise ValueError(f"shell candidate {self.key!r} has no argv")
        if self.surface == "browser" and self.element_ref is None and self.provenance == "dom":
            raise ValueError(f"DOM candidate {self.key!r} has no element reference")


@dataclass(frozen=True)
class ActionOutcome:
    """What happened when an action ran."""

    candidate: Candidate
    ok: bool
    observation: str
    exit_code: int | None = None
    refused_reason: str | None = None

    @property
    def refused(self) -> bool:
        return self.refused_reason is not None

    def history_entry(self) -> str:
        """One line for the PROGRESS section of the next state string."""
        if self.refused:
            return f"{self.candidate.description} -- REFUSED: {self.refused_reason}"
        status = "succeeded" if self.ok else f"failed (exit {self.exit_code})"
        return f"{self.candidate.description} -- {status}"
