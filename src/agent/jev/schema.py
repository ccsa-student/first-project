"""Wire types for the Jev System One API.

Shapes here were established by probing the live API, not from documentation.
The notable finding is that ``criteria`` is typed differently per question type:

    choice -> {option_name: description}   (max 255 options)
    noul   -> {"true": ..., "false": ...}
    score  -> [rung_description, ...]      (ordered, low to high)

Mirroring the server's own validation locally lets us reject a malformed
request before it costs a round trip, and turns its 422 ``loc`` paths into
something we can assert on.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# The server rejects more than this with a 400 naming the limit.
MAX_CHOICES = 255

# Established by bisection: 32,421 input tokens accepted, ~95k chars refused
# with max_tokens_exceeded. The cap covers the whole request, criteria included.
MAX_INPUT_TOKENS = 32_768


class ChoiceQuestion(BaseModel):
    """Pick one option. Returns a distribution over the options supplied.

    Confidence measures sharpness *among these options*. It says nothing about
    whether the right answer is among them -- see the gate panel for that.
    """

    type: Literal["choice"] = "choice"
    criteria: dict[str, str]

    @field_validator("criteria")
    @classmethod
    def _check_options(cls, v: dict[str, str]) -> dict[str, str]:
        if len(v) > MAX_CHOICES:
            raise ValueError(f"choice takes at most {MAX_CHOICES} options, got {len(v)}")
        if len(v) < 2:
            # A single-option choice returns confidence 1.0 trivially, which
            # reads downstream as certainty. Never let one reach the wire.
            raise ValueError("choice needs at least 2 options to be meaningful")
        descriptions = list(v.values())
        if len(set(descriptions)) != len(descriptions):
            # Duplicate descriptions split probability mass and collapse
            # confidence, which the gate would misread as "stuck".
            raise ValueError("choice option descriptions must be unique")
        return v


class NoulQuestion(BaseModel):
    """A single 0..1 judgement. Returns a bare float with no confidence field.

    The value is not a calibrated probability: a perfectly healthy page scored
    0.62 on an "is this actionable" noul. Thresholds are fitted per question.
    """

    type: Literal["noul"] = "noul"
    criteria: dict[Literal["true", "false"], str]

    @field_validator("criteria")
    @classmethod
    def _check_poles(cls, v: dict[str, str]) -> dict[str, str]:
        if set(v) != {"true", "false"}:
            raise ValueError("noul criteria needs exactly 'true' and 'false' keys")
        return v


class ScoreQuestion(BaseModel):
    """Position on an ordered ladder. Returns a fractional expected value."""

    type: Literal["score"] = "score"
    criteria: list[str]

    @field_validator("criteria")
    @classmethod
    def _check_rungs(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("score needs at least 2 rungs")
        return v


Question = Annotated[
    Union[ChoiceQuestion, NoulQuestion, ScoreQuestion],
    Field(discriminator="type"),
]


class JevRequest(BaseModel):
    model: str = "jev-latest"
    state: str
    questions: dict[str, Question]

    @model_validator(mode="after")
    def _check_request(self) -> JevRequest:
        if not self.questions:
            raise ValueError("at least one question is required")
        if not self.state.strip():
            # An empty state does not error server-side; it returns a
            # plausible-looking 0.45. Refuse it here instead.
            raise ValueError("state must not be empty")
        return self


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    confidence: float
    legend: dict[str, str]
    probabilities: dict[str, float]


Answer = Annotated[
    Union[ChoiceAnswer, NoulAnswer, ScoreAnswer],
    Field(discriminator="type"),
]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class JevResponse(BaseModel):
    # The concrete served version, e.g. "jev-1.13.0". Thresholds are fitted
    # against a specific version, so this is recorded and checked for drift.
    model: str
    answers: dict[str, Answer]
    usage: Usage

    def choice(self, key: str) -> ChoiceAnswer:
        answer = self.answers[key]
        if not isinstance(answer, ChoiceAnswer):
            raise TypeError(f"question {key!r} answered as {answer.type}, not choice")
        return answer

    def noul(self, key: str) -> float:
        answer = self.answers[key]
        if not isinstance(answer, NoulAnswer):
            raise TypeError(f"question {key!r} answered as {answer.type}, not noul")
        return answer.noul

    def score(self, key: str) -> ScoreAnswer:
        answer = self.answers[key]
        if not isinstance(answer, ScoreAnswer):
            raise TypeError(f"question {key!r} answered as {answer.type}, not score")
        return answer
