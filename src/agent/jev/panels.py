"""Builds the question sets for each Jev call.

Batching is free -- latency was flat from 1 to 500 questions, and a realistic
255-option payload with 20 nouls returned in 0.34s -- so a step is a small
number of large requests rather than many small ones.

    Call A  assess + select   (gate panel, progress ladder, action choice)
    Call A' re-select         (same, with LLM proposals appended)
    Call B  guard             (irreversibility + precondition on the choice)

The binding constraint is tokens, not questions: the 32,768 ceiling covers
question criteria too, and 255 options at ~140 chars is already 16k tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

from .criteria import (
    GATE_PANEL,
    GUARD_PANEL,
    GateQuestion,
    progress_question,
)
from .schema import ChoiceQuestion, NoulAnswer, Question
from .schema import JevResponse

ACTION_KEY = "act"
PROGRESS_KEY = "progress"


@dataclass(frozen=True)
class PanelResult:
    """A gate panel's answers, already aggregated across wording variants.

    The policy sees this, never raw per-variant values, so it cannot
    accidentally depend on one phrasing.
    """

    gate: dict[str, float]
    progress: float
    progress_confidence: float
    choice: str | None
    choice_confidence: float
    probabilities: dict[str, float]
    served_model: str


def build_assess_panel(candidates: dict[str, str]) -> dict[str, Question]:
    """Call A: gate panel + progress ladder + the action choice.

    ``candidates`` maps opaque option keys to their descriptions. Uniqueness
    and the 255 ceiling are enforced by ChoiceQuestion.
    """
    questions: dict[str, Question] = {}
    for gate_question in GATE_PANEL:
        questions.update(gate_question.questions())
    questions[PROGRESS_KEY] = progress_question()
    questions[ACTION_KEY] = ChoiceQuestion(criteria=candidates)
    return questions


def build_guard_panel() -> dict[str, Question]:
    """Call B: asked about the already-chosen action.

    The chosen action's description goes into the state, not into the
    criteria, so these stay frozen registry strings.
    """
    questions: dict[str, Question] = {}
    for gate_question in GUARD_PANEL:
        questions.update(gate_question.questions())
    return questions


def aggregate(
    response: JevResponse,
    panel: tuple[GateQuestion, ...] = GATE_PANEL,
    include_action: bool = True,
) -> PanelResult:
    """Collapse per-variant answers into one value per gate signal."""
    raw: dict[str, float] = {
        key: answer.noul
        for key, answer in response.answers.items()
        if isinstance(answer, NoulAnswer)
    }
    gate = {question.name: question.aggregate(raw) for question in panel}

    progress = 0.0
    progress_confidence = 0.0
    if PROGRESS_KEY in response.answers:
        score = response.score(PROGRESS_KEY)
        progress = score.score
        progress_confidence = score.confidence

    choice = None
    choice_confidence = 0.0
    probabilities: dict[str, float] = {}
    if include_action and ACTION_KEY in response.answers:
        answer = response.choice(ACTION_KEY)
        choice = answer.choice
        choice_confidence = answer.confidence
        probabilities = answer.probabilities

    return PanelResult(
        gate=gate,
        progress=progress,
        progress_confidence=progress_confidence,
        choice=choice,
        choice_confidence=choice_confidence,
        probabilities=probabilities,
        served_model=response.model,
    )


def spread(question_name: str, response: JevResponse) -> float:
    """Max-min across a question's wording variants.

    A wide spread means the variants disagree, which is the signal that a
    question's wording is doing more work than its subject. The tuner reports
    this alongside band population.
    """
    values = [
        answer.noul
        for key, answer in response.answers.items()
        if key.startswith(f"{question_name}__v") and isinstance(answer, NoulAnswer)
    ]
    return max(values) - min(values) if len(values) > 1 else 0.0
