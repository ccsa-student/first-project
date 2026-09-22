"""Structured step records.

The trace is the run's evidence: what the panel said, which rule fired, what
was chosen, where it came from, and what it cost. The dashboard streams it,
`record-from-run` turns halted steps into fixtures, and the metrics read it.

Recording the rule that fired matters as much as recording the decision.
"Halted" is not useful; "halted because target_present 0.34 fell below 0.44"
is what tells you whether the threshold or the candidate list was at fault.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..act.actions import ActionOutcome
from .policy import StepDecision


@dataclass
class StepRecord:
    index: int
    verdict: str
    reason: str
    gate: dict[str, float]
    progress: float
    choice: str | None
    choice_description: str | None
    choice_confidence: float
    provenance: str | None
    candidate_count: int
    observation: str | None = None
    exit_code: int | None = None
    refused_reason: str | None = None
    jev_input_tokens: int = 0
    jev_calls: int = 0
    llm_calls: int = 0
    llm_cost_usd: float = 0.0
    served_model: str = ""


@dataclass
class RunTrace:
    task: str
    params: dict[str, str] = field(default_factory=dict)
    steps: list[StepRecord] = field(default_factory=list)
    outcome: str = "running"
    thresholds_hash: str = ""
    jev_model: str = ""

    def add(self, record: StepRecord) -> None:
        self.steps.append(record)

    @property
    def jev_calls(self) -> int:
        return sum(s.jev_calls for s in self.steps)

    @property
    def llm_calls(self) -> int:
        return sum(s.llm_calls for s in self.steps)

    @property
    def jev_input_tokens(self) -> int:
        return sum(s.jev_input_tokens for s in self.steps)

    @property
    def llm_cost_usd(self) -> float:
        return sum(s.llm_cost_usd for s in self.steps)

    def llm_per_hundred_steps(self) -> float:
        """The headline metric: how much of the work Jev carries unaided."""
        return (self.llm_calls / len(self.steps) * 100) if self.steps else 0.0

    def summary(self) -> str:
        lines = [
            f"outcome: {self.outcome}",
            f"steps: {len(self.steps)}",
            f"Jev calls: {self.jev_calls} ({self.jev_input_tokens:,} input tokens)",
            f"LLM calls: {self.llm_calls} "
            f"({self.llm_per_hundred_steps():.0f} per 100 steps, "
            f"${self.llm_cost_usd:.4f})",
        ]
        if self.steps:
            lines.append(f"final rule: {self.steps[-1].reason}")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "params": self.params,
            "outcome": self.outcome,
            "thresholds_hash": self.thresholds_hash,
            "jev_model": self.jev_model,
            "totals": {
                "steps": len(self.steps),
                "jev_calls": self.jev_calls,
                "jev_input_tokens": self.jev_input_tokens,
                "llm_calls": self.llm_calls,
                "llm_per_100_steps": round(self.llm_per_hundred_steps(), 1),
                "llm_cost_usd": round(self.llm_cost_usd, 6),
            },
            "steps": [asdict(s) for s in self.steps],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        return path


def record_step(
    index: int,
    decision: StepDecision,
    *,
    choice_description: str | None,
    choice_confidence: float,
    provenance: str | None,
    candidate_count: int,
    jev_input_tokens: int,
    jev_calls: int,
    served_model: str,
    outcome: ActionOutcome | None = None,
) -> StepRecord:
    return StepRecord(
        index=index,
        verdict=decision.verdict,
        reason=decision.reason,
        gate=dict(decision.gate),
        progress=decision.progress,
        choice=decision.action_key,
        choice_description=choice_description,
        choice_confidence=choice_confidence,
        provenance=provenance,
        candidate_count=candidate_count,
        observation=outcome.observation if outcome else None,
        exit_code=outcome.exit_code if outcome else None,
        refused_reason=outcome.refused_reason if outcome else None,
        jev_input_tokens=jev_input_tokens,
        jev_calls=jev_calls,
        served_model=served_model,
    )
