"""The decision policy: pure threshold logic over a panel result.

Deliberately free of I/O, network, browser and LLM. Everything the agent
decides happens here, so everything the agent decides is unit-testable
without spending money or launching a browser.

Rule order matters and is not arbitrary:

    1. error / looping        -> HALT      (state is unusable; nothing else applies)
    2. target_present low     -> PROPOSE   (the answer is not in the list; ask the LLM)
       still low after retry  -> HALT
    3. complete               -> SUCCEED   (checked before acting, or we act past done)
    4. otherwise                           -> take the choice winner
    5. no value for a field   -> GENERATE_VALUE
    6. irreversible           -> CONFIRM   (before executing, never after)
    7. precondition failed    -> REOBSERVE

The ordering that matters most is 2 before 4. Choice confidence cannot detect
an unfit option set -- measured 0.96 confidence on a wrong answer when the
right one had been pruned away -- so the gate must intercept before the
winner is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..jev.panels import PanelResult

Verdict = Literal[
    "act",
    "halt",
    "succeed",
    "confirm",
    "reobserve",
    "propose",
    "generate_value",
]


@dataclass(frozen=True)
class Thresholds:
    """Cut-points. Fitted by calib.tune, never authored by hand.

    The defaults here are the provisional values from the initial API probe.
    They exist so the policy is runnable before the tuner has produced a
    fitted set; `Thresholds.require_fitted()` refuses them in a real run.
    """

    target_present: float = 0.50
    is_error: float = 0.60
    looping: float = 0.70
    injection: float = 0.60
    irreversible: float = 0.40
    precondition: float = 0.50
    # Measured: a finished task scored 2.00 at confidence 1.00, while a
    # mid-task state scored 1.79 at 0.68. A cut at 1.70/0.60 would call that
    # mid-task state complete -- a false success, which is the worst failure
    # this policy can produce. Provisional cuts therefore sit above the
    # measured mid-task point on both axes, and the tuner must fit them
    # against fixtures that include near-complete states.
    complete: float = 1.90
    complete_confidence: float = 0.85
    choice_confidence: float = 0.50

    fitted: bool = False
    registry_hash: str | None = None
    jev_model: str | None = None

    def require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(
                "refusing to run on provisional thresholds; "
                "run `uv run python -m agent.calib.tune` first"
            )


@dataclass(frozen=True)
class StepDecision:
    verdict: Verdict
    reason: str
    action_key: str | None = None
    gate: dict[str, float] = field(default_factory=dict)
    progress: float = 0.0
    # Set when the decision was reached after a Proposer round, so the
    # controller does not loop on it forever.
    after_proposal: bool = False


def decide(
    panel: PanelResult,
    thresholds: Thresholds,
    *,
    after_proposal: bool = False,
    llm_available: bool = True,
) -> StepDecision:
    """Map a panel result onto exactly one verdict.

    ``after_proposal`` marks a re-assessment following a Proposer call, which
    turns a second target_present failure into a halt rather than another
    proposal round.

    ``llm_available`` is False when no LLM is configured; the Proposer branch
    then degrades to a halt instead of asking for something impossible.
    """
    gate = panel.gate

    # ---- 1. unusable state -------------------------------------------------
    if gate.get("is_error", 0.0) >= thresholds.is_error:
        return StepDecision(
            verdict="halt",
            reason=(
                f"is_error {gate['is_error']:.2f} >= {thresholds.is_error:.2f}: "
                "the page or last command is in a failure state"
            ),
            gate=gate,
            progress=panel.progress,
            after_proposal=after_proposal,
        )

    if gate.get("looping", 0.0) >= thresholds.looping:
        return StepDecision(
            verdict="halt",
            reason=(
                f"looping {gate['looping']:.2f} >= {thresholds.looping:.2f}: "
                "repeating an ineffective action without progressing"
            ),
            gate=gate,
            progress=panel.progress,
            after_proposal=after_proposal,
        )

    # ---- 2. the answer is not in the list ----------------------------------
    # This is checked BEFORE trusting the choice winner, because confidence
    # cannot see this failure. Measured: 0.96 confidence on a wrong option
    # when the correct one had been pruned out.
    if gate.get("target_present", 1.0) < thresholds.target_present:
        if after_proposal or not llm_available:
            detail = (
                "the Proposer's candidates did not help"
                if after_proposal
                else "no LLM is configured to propose alternatives"
            )
            return StepDecision(
                verdict="halt",
                reason=(
                    f"target_present {gate['target_present']:.2f} < "
                    f"{thresholds.target_present:.2f}: {detail}"
                ),
                gate=gate,
                progress=panel.progress,
                after_proposal=after_proposal,
            )
        return StepDecision(
            verdict="propose",
            reason=(
                f"target_present {gate['target_present']:.2f} < "
                f"{thresholds.target_present:.2f}: the needed action is absent "
                "from the candidate list"
            ),
            gate=gate,
            progress=panel.progress,
        )

    # ---- 3. done -----------------------------------------------------------
    if (
        panel.progress >= thresholds.complete
        and panel.progress_confidence >= thresholds.complete_confidence
    ):
        return StepDecision(
            verdict="succeed",
            reason=(
                f"progress {panel.progress:.2f} >= {thresholds.complete:.2f} "
                f"at confidence {panel.progress_confidence:.2f}"
            ),
            gate=gate,
            progress=panel.progress,
            after_proposal=after_proposal,
        )

    # ---- 4. act ------------------------------------------------------------
    if panel.choice is None:
        return StepDecision(
            verdict="halt",
            reason="no action was selected",
            gate=gate,
            progress=panel.progress,
            after_proposal=after_proposal,
        )

    if panel.choice_confidence < thresholds.choice_confidence:
        # Reached only when target_present passed, so the answer is believed
        # present and the model is genuinely torn between options.
        return StepDecision(
            verdict="halt",
            reason=(
                f"choice confidence {panel.choice_confidence:.2f} < "
                f"{thresholds.choice_confidence:.2f} despite the target being "
                "present: genuinely ambiguous"
            ),
            gate=gate,
            progress=panel.progress,
            after_proposal=after_proposal,
        )

    return StepDecision(
        verdict="act",
        reason=f"selected at confidence {panel.choice_confidence:.2f}",
        action_key=panel.choice,
        gate=gate,
        progress=panel.progress,
        after_proposal=after_proposal,
    )


def guard(
    action_key: str,
    guard_gate: dict[str, float],
    thresholds: Thresholds,
) -> StepDecision:
    """Second-call check on an already-selected action.

    Precondition is checked before irreversibility: if the element is gone,
    the irreversibility answer refers to something that no longer exists.
    """
    if guard_gate.get("precondition", 1.0) < thresholds.precondition:
        return StepDecision(
            verdict="reobserve",
            reason=(
                f"precondition {guard_gate['precondition']:.2f} < "
                f"{thresholds.precondition:.2f}: the target is no longer present"
            ),
            action_key=action_key,
            gate=guard_gate,
        )

    if guard_gate.get("irreversible", 0.0) >= thresholds.irreversible:
        return StepDecision(
            verdict="confirm",
            reason=(
                f"irreversible {guard_gate['irreversible']:.2f} >= "
                f"{thresholds.irreversible:.2f}: needs human confirmation"
            ),
            action_key=action_key,
            gate=guard_gate,
        )

    return StepDecision(
        verdict="act",
        reason="cleared the guard panel",
        action_key=action_key,
        gate=guard_gate,
    )
