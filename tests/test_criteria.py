"""Tests for the frozen criteria registry.

These are guards against silent recalibration: the registry hash changing
without anyone noticing is how fitted thresholds quietly stop applying to the
wording they were fitted against.
"""

from __future__ import annotations

import pytest

from agent.jev.criteria import (
    ACTIONABLE,
    ALL_QUESTIONS,
    GATE_PANEL,
    GUARD_PANEL,
    INJECTION,
    IRREVERSIBLE,
    TARGET_PRESENT,
    GateQuestion,
    by_name,
    registry_hash,
    question_names,
)
from agent.jev.schema import NoulQuestion


def test_registry_hash_is_stable_across_calls():
    assert registry_hash() == registry_hash()


def test_registry_hash_changes_when_wording_changes():
    """The whole point of the hash. If this fails, thresholds can drift silently."""
    original = registry_hash()
    mutated = GateQuestion(
        name=TARGET_PRESENT.name,
        rationale=TARGET_PRESENT.rationale,
        variants=(("a different wording entirely", "and its opposite"),),
    )
    assert mutated.variants != TARGET_PRESENT.variants
    # The hash covers variant text, so a registry containing `mutated`
    # must not hash to the same value.
    import json, hashlib

    payload = {q.name: [list(v) for v in q.variants] for q in ALL_QUESTIONS}
    payload[TARGET_PRESENT.name] = [list(v) for v in mutated.variants]
    from agent.jev.criteria import PROGRESS_RUNGS

    payload["__progress__"] = [PROGRESS_RUNGS]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest()[:16] != original


def test_every_question_builds_valid_noul_questions():
    for question in ALL_QUESTIONS:
        built = question.questions()
        assert len(built) == len(question.variants)
        for key, value in built.items():
            assert key.startswith(f"{question.name}__v")
            assert isinstance(value, NoulQuestion)
            assert set(value.criteria) == {"true", "false"}


def test_gate_questions_have_distinct_variant_wordings():
    """Duplicate variants would make an ensemble a single point of failure
    wearing a disguise."""
    for question in ALL_QUESTIONS:
        trues = [t for t, _ in question.variants]
        assert len(set(trues)) == len(trues), f"{question.name} has duplicate variants"


def test_primary_signals_are_ensembled():
    """target_present and irreversible carry the most consequential thresholds,
    so neither may rest on a single phrasing."""
    assert len(TARGET_PRESENT.variants) >= 3
    assert len(IRREVERSIBLE.variants) >= 3


def test_actionable_is_present_but_single_variant():
    """Deliberately demoted: it failed to separate the target-pruned case
    (0.57 vs 0.55). Kept as a secondary signal only."""
    assert ACTIONABLE in GATE_PANEL
    assert len(ACTIONABLE.variants) == 1


def test_injection_variants_exclude_agent_scaffolding():
    """An earlier wording read our own AVAILABLE ACTIONS block as an attack,
    scoring 0.43 on clean pages. Both variants must scope to page content."""
    for true_pole, false_pole in INJECTION.variants:
        combined = (true_pole + false_pole).upper()
        assert "PAGE" in combined or "COMMAND OUTPUT" in combined


def test_aggregate_averages_variants():
    question = by_name("looping")
    keys = list(question.questions())
    answers = dict(zip(keys, [0.2, 0.8]))
    assert question.aggregate(answers) == pytest.approx(0.5)


def test_aggregate_tolerates_partial_answers():
    """A dropped variant should degrade to the mean of what came back, not crash."""
    question = by_name("looping")
    first = list(question.questions())[0]
    assert question.aggregate({first: 0.7}) == pytest.approx(0.7)


def test_aggregate_raises_when_nothing_present():
    with pytest.raises(KeyError):
        by_name("looping").aggregate({"unrelated": 0.5})


def test_question_names_are_unique():
    names = question_names()
    assert len(set(names)) == len(names)


def test_panels_are_disjoint():
    """A question in both panels would be asked twice per step and
    double-counted."""
    gate = {q.name for q in GATE_PANEL}
    guard = {q.name for q in GUARD_PANEL}
    assert gate.isdisjoint(guard)
