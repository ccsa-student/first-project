"""Fixture format and loader.

A fixture is a labelled situation: a task, a history, an observed state, a
candidate action list, and the ground truth about what should happen. The
harness renders it into a Jev state string, asks the panel, and compares the
answer to the labels.

Two design choices worth stating, because they are what make the harness
survive changes to the code it tests:

1. Fixtures store the candidate list *as authored*, and labels reference
   candidates by key rather than by index or description. Rewording a
   description does not invalidate a label.

2. Fixtures record the labelled truth about whether the correct action is
   present at all (``target_present``). That is the signal the whole
   architecture turns on, and it is invisible to Jev's confidence, so the
   harness has to know it independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

FixtureClass = Literal[
    "healthy",
    "blocked",
    "error",
    "loading",
    "login_wall",
    "ambiguous",
    "target_pruned",
    "irreversible",
    "complete",
    "empty",
    "injected",
    "shell_clean",
    "shell_failed",
]

Surface = Literal["browser", "shell"]

FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "fixtures"


@dataclass(frozen=True)
class Labels:
    """Ground truth, authored by a human.

    ``correct`` is a set rather than a single key because ties are real: on
    many pages more than one next action is equally right.
    """

    correct: tuple[str, ...] = ()
    acceptable: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    irreversible: tuple[str, ...] = ()

    # Gate ground truth. None means "not labelled for this fixture", which the
    # metrics treat as "do not score", not as False.
    target_present: bool | None = None
    is_error: bool | None = None
    looping: bool | None = None
    injection: bool | None = None
    complete: bool | None = None

    def gate_truth(self) -> dict[str, bool]:
        return {
            name: value
            for name, value in (
                ("target_present", self.target_present),
                ("is_error", self.is_error),
                ("looping", self.looping),
                ("injection", self.injection),
            )
            if value is not None
        }


@dataclass(frozen=True)
class Fixture:
    id: str
    fixture_class: FixtureClass
    surface: Surface
    task: str
    observation: str
    candidates: dict[str, str]
    labels: Labels
    params: dict[str, str] = field(default_factory=dict)
    history: tuple[str, ...] = ()
    notes: str = ""

    def state(self) -> str:
        """Render the Jev state string.

        Section order and headings are part of the calibrated surface: the
        criteria refer to TASK, PAGE, SHELL and AVAILABLE ACTIONS by name, so
        changing these headings is a recalibration.
        """
        parts = [f"## TASK\n{self.task}"]

        if self.params:
            bound = "\n".join(f"  {k} = {v}" for k, v in sorted(self.params.items()))
            parts.append(f"## PARAMETERS\n{bound}")

        if self.history:
            steps = "\n".join(
                f"  {i + 1}. {entry}" for i, entry in enumerate(self.history)
            )
            parts.append(f"## PROGRESS\nstep {len(self.history) + 1}. Recent actions:\n{steps}")
        else:
            parts.append("## PROGRESS\nstep 1. No actions taken yet.")

        heading = "PAGE" if self.surface == "browser" else "SHELL"
        parts.append(f"## {heading}\n{self.observation}")

        actions = "\n".join(f"  - {desc}" for desc in self.candidates.values())
        parts.append(f"## AVAILABLE ACTIONS\n{actions}")

        return "\n".join(parts)

    def validate(self) -> list[str]:
        """Return problems with this fixture's own consistency.

        Authoring mistakes here are worse than code bugs: they silently
        corrupt every threshold fitted against them.
        """
        problems: list[str] = []
        keys = set(self.candidates)

        for group, label in (
            (self.labels.correct, "correct"),
            (self.labels.acceptable, "acceptable"),
            (self.labels.forbidden, "forbidden"),
            (self.labels.irreversible, "irreversible"),
        ):
            unknown = set(group) - keys
            if unknown:
                problems.append(f"{label} references unknown candidates: {sorted(unknown)}")

        overlap = set(self.labels.correct) & set(self.labels.forbidden)
        if overlap:
            problems.append(f"candidates labelled both correct and forbidden: {sorted(overlap)}")

        # target_present means "a usable action is in the list", which an
        # ambiguous fixture satisfies with `acceptable` alone -- there is no
        # single right answer on a cookie dialog, but the answer is present.
        if self.labels.target_present is True and not (
            self.labels.correct or self.labels.acceptable
        ):
            problems.append(
                "target_present is True but no correct or acceptable candidate is named"
            )

        if self.labels.target_present is False and (
            self.labels.correct or self.labels.acceptable
        ):
            problems.append(
                "target_present is False but correct/acceptable candidates are named"
            )

        if len(self.candidates) < 2:
            problems.append("needs at least 2 candidates for a meaningful choice")

        descriptions = list(self.candidates.values())
        if len(set(descriptions)) != len(descriptions):
            problems.append("duplicate candidate descriptions split probability mass")

        return problems


def _to_json(fixture: Fixture) -> dict:
    return {
        "id": fixture.id,
        "class": fixture.fixture_class,
        "surface": fixture.surface,
        "task": fixture.task,
        "params": fixture.params,
        "history": list(fixture.history),
        "observation": fixture.observation,
        "candidates": fixture.candidates,
        "notes": fixture.notes,
        "labels": {
            "correct": list(fixture.labels.correct),
            "acceptable": list(fixture.labels.acceptable),
            "forbidden": list(fixture.labels.forbidden),
            "irreversible": list(fixture.labels.irreversible),
            "target_present": fixture.labels.target_present,
            "is_error": fixture.labels.is_error,
            "looping": fixture.labels.looping,
            "injection": fixture.labels.injection,
            "complete": fixture.labels.complete,
        },
    }


def _from_json(payload: dict) -> Fixture:
    raw = payload["labels"]
    return Fixture(
        id=payload["id"],
        fixture_class=payload["class"],
        surface=payload["surface"],
        task=payload["task"],
        params=payload.get("params", {}),
        history=tuple(payload.get("history", [])),
        observation=payload["observation"],
        candidates=payload["candidates"],
        notes=payload.get("notes", ""),
        labels=Labels(
            correct=tuple(raw.get("correct", [])),
            acceptable=tuple(raw.get("acceptable", [])),
            forbidden=tuple(raw.get("forbidden", [])),
            irreversible=tuple(raw.get("irreversible", [])),
            target_present=raw.get("target_present"),
            is_error=raw.get("is_error"),
            looping=raw.get("looping"),
            injection=raw.get("injection"),
            complete=raw.get("complete"),
        ),
    )


def save(fixture: Fixture, root: Path = FIXTURE_ROOT) -> Path:
    directory = root / fixture.id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "fixture.json"
    path.write_text(json.dumps(_to_json(fixture), indent=2) + "\n")
    return path


def load(path: Path) -> Fixture:
    return _from_json(json.loads(path.read_text()))


def load_all(root: Path = FIXTURE_ROOT) -> list[Fixture]:
    if not root.exists():
        return []
    fixtures = [load(p) for p in sorted(root.glob("*/fixture.json"))]
    broken = [(f.id, f.validate()) for f in fixtures]
    problems = [(fid, probs) for fid, probs in broken if probs]
    if problems:
        detail = "\n".join(f"  {fid}: {'; '.join(probs)}" for fid, probs in problems)
        raise ValueError(f"inconsistent fixtures:\n{detail}")
    return fixtures


def cassette_path(fixture: Fixture, root: Path = FIXTURE_ROOT) -> Path:
    return root / fixture.id / "cassette.jsonl"
