"""Tests for the decision policy.

Every case here is built from a measured value where one exists, so the tests
encode the empirical findings rather than invented numbers.
"""

from __future__ import annotations

import pytest

from agent.jev.panels import PanelResult
from agent.loop.policy import StepDecision, Thresholds, decide, guard

T = Thresholds()


def panel(
    *,
    target_present=0.85,
    is_error=0.15,
    looping=0.18,
    injection=0.30,
    actionable=0.65,
    progress=1.0,
    progress_confidence=0.70,
    choice="opt_1",
    choice_confidence=0.95,
) -> PanelResult:
    """A healthy mid-task panel, with knobs for each signal."""
    return PanelResult(
        gate={
            "target_present": target_present,
            "is_error": is_error,
            "looping": looping,
            "injection": injection,
            "actionable": actionable,
        },
        progress=progress,
        progress_confidence=progress_confidence,
        choice=choice,
        choice_confidence=choice_confidence,
        probabilities={"opt_1": choice_confidence},
        served_model="jev-1.13.0",
    )


# --------------------------------------------------------------------------
# The ordinary path
# --------------------------------------------------------------------------


def test_healthy_panel_acts():
    decision = decide(panel(), T)
    assert decision.verdict == "act"
    assert decision.action_key == "opt_1"


def test_completed_task_succeeds():
    # Measured on a finished task: score 2.00 at confidence 1.00.
    decision = decide(panel(progress=2.0, progress_confidence=1.0), T)
    assert decision.verdict == "succeed"


def test_measured_mid_task_state_does_not_report_success():
    """The false-success guard.

    Measured mid-task: progress 1.79 at confidence 0.68. Measured complete:
    2.00 at 1.00. These sit close enough that a careless cut declares the
    mid-task state finished, which is the worst failure the policy can
    produce -- the agent stops and reports done with the job half done.
    """
    decision = decide(panel(progress=1.79, progress_confidence=0.68), T)
    assert decision.verdict != "succeed"


def test_measured_complete_state_still_succeeds():
    """The guard above must not be so tight that nothing ever completes."""
    decision = decide(panel(progress=2.0, progress_confidence=1.0), T)
    assert decision.verdict == "succeed"


def test_partial_progress_keeps_acting():
    decision = decide(panel(progress=1.08, progress_confidence=0.55), T)
    assert decision.verdict == "act"


# --------------------------------------------------------------------------
# Rule 1: unusable state
# --------------------------------------------------------------------------


def test_error_page_halts():
    # Measured on a 403 block page: is_error 0.94.
    decision = decide(panel(is_error=0.94), T)
    assert decision.verdict == "halt"
    assert "is_error" in decision.reason


def test_looping_halts():
    # Measured on a history repeating one action five times: looping 0.88.
    decision = decide(panel(looping=0.88), T)
    assert decision.verdict == "halt"
    assert "looping" in decision.reason


def test_error_outranks_everything_else():
    """A broken page with a confident choice must still halt."""
    decision = decide(panel(is_error=0.94, choice_confidence=1.0, progress=2.0), T)
    assert decision.verdict == "halt"


# --------------------------------------------------------------------------
# Rule 2: the finding that drove the architecture
# --------------------------------------------------------------------------


def test_target_absent_proposes_even_at_high_confidence():
    """The decisive case. Measured: with the correct target pruned away, Jev
    returned a wrong option at 0.96 confidence while target_present fell to
    0.19. Confidence must not be allowed to override the gate."""
    decision = decide(panel(target_present=0.19, choice_confidence=0.96), T)
    assert decision.verdict == "propose"
    assert decision.action_key is None


def test_target_absent_halts_after_a_failed_proposal():
    decision = decide(
        panel(target_present=0.19, choice_confidence=0.96), T, after_proposal=True
    )
    assert decision.verdict == "halt"
    assert "Proposer" in decision.reason


def test_target_absent_halts_when_no_llm_configured():
    decision = decide(
        panel(target_present=0.19), T, llm_available=False
    )
    assert decision.verdict == "halt"
    assert "no LLM" in decision.reason


def test_target_present_check_precedes_completion():
    """A missing target must not be masked by an apparently complete score."""
    decision = decide(panel(target_present=0.19, progress=2.0, progress_confidence=1.0), T)
    assert decision.verdict == "propose"


def test_error_precedes_proposal():
    """A blocked page should halt, not spend money on a Proposer call."""
    decision = decide(panel(is_error=0.94, target_present=0.19), T)
    assert decision.verdict == "halt"
    assert "is_error" in decision.reason


# --------------------------------------------------------------------------
# Rule 4: ambiguity
# --------------------------------------------------------------------------


def test_low_confidence_with_target_present_halts_as_ambiguous():
    decision = decide(panel(target_present=0.85, choice_confidence=0.38), T)
    assert decision.verdict == "halt"
    assert "ambiguous" in decision.reason


def test_missing_choice_halts():
    decision = decide(panel(choice=None), T)
    assert decision.verdict == "halt"


# --------------------------------------------------------------------------
# The guard panel
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.85, "confirm"),  # click_buy
        (0.73, "confirm"),  # delete account
        (0.66, "confirm"),  # rm -rf ./build
        (0.60, "confirm"),  # confirm unsubscribe
        (0.19, "act"),      # browser back
        (0.16, "act"),      # ls -la
        (0.10, "act"),      # scroll
    ],
)
def test_irreversibility_gate_matches_measured_values(value, expected):
    """Every value here was measured. The cut at 0.40 sits in the empty band
    between 0.19 and 0.60."""
    decision = guard("opt_1", {"irreversible": value, "precondition": 0.9}, T)
    assert decision.verdict == expected


def test_stale_target_reobserves_before_confirming():
    """Precondition is checked first: an irreversibility answer about a
    vanished element is meaningless."""
    decision = guard("opt_1", {"irreversible": 0.85, "precondition": 0.1}, T)
    assert decision.verdict == "reobserve"


def test_guard_passes_action_key_through():
    decision = guard("opt_7", {"irreversible": 0.1, "precondition": 0.9}, T)
    assert decision.verdict == "act"
    assert decision.action_key == "opt_7"


# --------------------------------------------------------------------------
# Threshold provenance
# --------------------------------------------------------------------------


def test_provisional_thresholds_refuse_a_real_run():
    with pytest.raises(RuntimeError, match="provisional"):
        Thresholds().require_fitted()


def test_fitted_thresholds_are_accepted():
    Thresholds(fitted=True, registry_hash="abc", jev_model="jev-1.13.0").require_fitted()


def test_decision_is_immutable():
    decision = decide(panel(), T)
    with pytest.raises(Exception):
        decision.verdict = "halt"  # type: ignore[misc]
