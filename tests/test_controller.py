"""Tests for the agent loop.

Driven by a scripted fake Jev client rather than cassettes, because what is
under test here is orchestration -- which verdict leads to which action --
not what Jev says. Scripting the answers lets each path be exercised in
isolation, including ones that are hard to provoke against the real API.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.act.actions import ActionOutcome, Candidate
from agent.act.executor import BrowserNotAvailable, Executor
from agent.jev.criteria import GATE_PANEL, GUARD_PANEL
from agent.jev.schema import JevResponse
from agent.loop.controller import (
    AutoDeny,
    Controller,
    ShellObserver,
    Supervisor,
    compose_state,
    load_thresholds,
)
from agent.loop.policy import StepDecision, Thresholds
from agent.tools.shell import Sandbox

T = Thresholds(fitted=True)


class ScriptedClient:
    """Answers whatever the panel asks, from a script of gate values."""

    def __init__(self, script: list[dict]) -> None:
        self.script = script
        self.calls = 0
        self.step_index = 0
        self.last_model_served = "jev-fake"

    def ask(self, state: str, questions: dict) -> JevResponse:
        answers: dict = {}
        is_guard = any(k.startswith("precondition__") for k in questions)
        panel = GUARD_PANEL if is_guard else GATE_PANEL

        # One script entry per loop step, not per call. The assess call
        # consumes an entry; the guard call re-reads the one just consumed, so
        # both halves of a step answer from the same script. Advancing per
        # call instead would have the guard answering from the next step.
        self.calls += 1
        if is_guard:
            index = max(0, self.step_index - 1)
        else:
            index = self.step_index
            self.step_index += 1
        step = self.script[min(index, len(self.script) - 1)]

        # Defaults describe a healthy step, so a script only has to name the
        # signal it is exercising. A flat 0.5 default would sit above the
        # irreversibility cut and make every action prompt for confirmation.
        healthy = {
            "target_present": 0.9,
            "precondition": 0.9,
            "is_error": 0.1,
            "looping": 0.1,
            "injection": 0.1,
            "actionable": 0.8,
            "irreversible": 0.05,
        }
        for question in panel:
            value = step.get(question.name, healthy.get(question.name, 0.1))
            for i in range(len(question.variants)):
                answers[f"{question.name}__v{i}"] = {"type": "noul", "noul": value}

        if not is_guard:
            answers["progress"] = {
                "type": "score",
                "score": step.get("progress", 0.5),
                "confidence": step.get("progress_confidence", 0.9),
                "legend": {"0": "a", "1": "b", "2": "c"},
                "probabilities": {"0": 0.5, "1": 0.3, "2": 0.2},
            }
            choice_key = step.get("choice")
            option_keys = list(questions["act"].criteria)
            picked = choice_key if choice_key in option_keys else option_keys[0]
            answers["act"] = {
                "type": "choice",
                "choice": picked,
                "confidence": step.get("choice_confidence", 0.95),
                "probabilities": {picked: step.get("choice_confidence", 0.95)},
            }

        return JevResponse.model_validate(
            {
                "model": "jev-fake",
                "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 10},
            }
        )


class RecordingSupervisor(Supervisor):
    def __init__(self, approve: bool = False, takeover: str | None = None) -> None:
        self.approve = approve
        self.takeover = takeover
        self.confirmations: list[str] = []
        self.halts: list[str] = []

    def confirm(self, candidate: Candidate, reason: str) -> bool:
        self.confirmations.append(candidate.key)
        return self.approve

    def on_halt(self, decision, candidates):
        self.halts.append(decision.reason)
        return self.takeover


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# TODO: x\n")
    (tmp_path / "README.md").write_text("# Demo\n")
    return tmp_path


def build(project: Path, script: list[dict], supervisor=None, max_steps=3):
    return Controller(
        client=ScriptedClient(script),  # type: ignore[arg-type]
        executor=Executor(sandbox=Sandbox(root=project)),
        observer=ShellObserver(root=project, params={"pattern": "TODO"}),
        thresholds=T,
        supervisor=supervisor or AutoDeny(),
        max_steps=max_steps,
        llm_available=False,
    )


# --------------------------------------------------------------------------
# State composition
# --------------------------------------------------------------------------


def test_state_always_contains_available_actions():
    """target_present asks whether the needed action appears in this section,
    so omitting it would silently disable the primary gate."""
    candidates = {
        "a": Candidate(key="a", description="do a", surface="shell", provenance="registry", argv=("ls",)),
        "b": Candidate(key="b", description="do b", surface="shell", provenance="registry", argv=("pwd",)),
    }
    state = compose_state("a task", {}, [], "an observation", candidates)
    assert "## AVAILABLE ACTIONS" in state
    assert "do a" in state and "do b" in state


def test_state_keeps_only_recent_history():
    candidates = {
        "a": Candidate(key="a", description="do a", surface="shell", provenance="registry", argv=("ls",)),
        "b": Candidate(key="b", description="do b", surface="shell", provenance="registry", argv=("pwd",)),
    }
    history = [f"step {i}" for i in range(20)]
    state = compose_state("t", {}, history, "obs", candidates)
    assert "step 19" in state
    assert "step 0" not in state


# --------------------------------------------------------------------------
# The ordinary path
# --------------------------------------------------------------------------


def test_acts_then_succeeds(project: Path):
    controller = build(
        project,
        [
            {"choice": "sh__ls__root", "progress": 0.5},
            {"progress": 2.0, "progress_confidence": 1.0},
        ],
    )
    trace = controller.run("list the directory")
    assert trace.outcome == "succeed"
    assert trace.steps[0].verdict == "act"
    assert trace.llm_calls == 0


def test_a_run_makes_two_jev_calls_per_acting_step(project: Path):
    """Call A to assess and select, Call B to guard."""
    controller = build(project, [{"choice": "sh__ls__root", "progress": 0.5}], max_steps=1)
    trace = controller.run("list")
    assert trace.steps[0].jev_calls == 2


def test_trace_records_the_rule_that_fired(project: Path):
    """"Halted" is not useful; the threshold comparison is."""
    controller = build(project, [{"target_present": 0.1}], max_steps=1)
    trace = controller.run("do something impossible")
    assert "target_present" in trace.steps[0].reason
    assert str(T.target_present) in trace.steps[0].reason


# --------------------------------------------------------------------------
# Halting and takeover
# --------------------------------------------------------------------------


def test_halts_when_the_target_is_absent(project: Path):
    supervisor = RecordingSupervisor()
    controller = build(project, [{"target_present": 0.1}], supervisor, max_steps=1)
    trace = controller.run("something no candidate covers")
    assert trace.outcome == "halt"
    assert supervisor.halts


def test_human_takeover_resumes_the_loop(project: Path):
    """A halt is recoverable, not terminal."""
    supervisor = RecordingSupervisor(takeover="sh__ls__root")
    controller = build(
        project,
        [
            {"target_present": 0.1},
            {"progress": 2.0, "progress_confidence": 1.0},
        ],
        supervisor,
        max_steps=3,
    )
    trace = controller.run("something")
    assert supervisor.halts
    assert trace.steps[0].provenance == "human"
    assert trace.outcome == "succeed"


def test_error_state_halts(project: Path):
    controller = build(project, [{"is_error": 0.95}], max_steps=1)
    assert controller.run("x").outcome == "halt"


def test_looping_halts(project: Path):
    controller = build(project, [{"looping": 0.95}], max_steps=1)
    assert controller.run("x").outcome == "halt"


# --------------------------------------------------------------------------
# The guard panel
# --------------------------------------------------------------------------


def test_irreversible_action_pauses_and_is_denied_by_default(project: Path):
    """AutoDeny is deliberately conservative: an unattended run should stop at
    an irreversible action rather than guess."""
    supervisor = RecordingSupervisor(approve=False)
    controller = build(
        project, [{"choice": "sh__ls__root", "irreversible": 0.95}], supervisor, max_steps=1
    )
    trace = controller.run("x")
    assert trace.outcome == "confirm_denied"
    assert supervisor.confirmations


def test_approved_irreversible_action_executes(project: Path):
    supervisor = RecordingSupervisor(approve=True)
    controller = build(
        project, [{"choice": "sh__ls__root", "irreversible": 0.95}], supervisor, max_steps=1
    )
    trace = controller.run("x")
    assert supervisor.confirmations
    assert trace.steps[0].verdict == "act"
    assert trace.steps[0].exit_code == 0


def test_declared_irreversible_template_confirms_despite_a_low_score(project: Path):
    """The author's declaration is not subject to the classifier's judgement.

    Found in a live run: `uv sync` declares itself irreversible but scores
    below the fitted cut, and the agent ran it with no prompt.
    """
    supervisor = RecordingSupervisor(approve=False)
    controller = build(
        project,
        [{"choice": "sh__uv_sync", "irreversible": 0.01}],
        supervisor,
        max_steps=1,
    )
    trace = controller.run("install dependencies")
    assert trace.outcome == "confirm_denied"
    assert "template declares" in trace.steps[0].reason


def test_stale_target_reobserves_without_acting(project: Path):
    controller = build(
        project,
        [
            {"choice": "sh__ls__root", "precondition": 0.05},
            {"progress": 2.0, "progress_confidence": 1.0},
        ],
        max_steps=3,
    )
    trace = controller.run("x")
    assert trace.steps[0].verdict == "reobserve"
    assert trace.steps[0].exit_code is None


# --------------------------------------------------------------------------
# Executor behaviour
# --------------------------------------------------------------------------


def test_a_refused_command_is_an_outcome_not_a_crash(project: Path):
    """Containment held; the loop should record it and carry on."""
    executor = Executor(sandbox=Sandbox(root=project))
    candidate = Candidate(
        key="bad", description="sudo something", surface="shell",
        provenance="llm", argv=("sudo", "ls"),
    )
    outcome = executor.execute(candidate)
    assert not outcome.ok
    assert outcome.refused
    assert "privilege" in outcome.refused_reason


def test_unbound_parameters_are_refused_before_execution(project: Path):
    executor = Executor(sandbox=Sandbox(root=project))
    candidate = Candidate(
        key="wc", description="count lines", surface="shell",
        provenance="registry", argv=("wc", "-l", "{file}"), unbound=("file",),
    )
    outcome = executor.execute(candidate)
    assert outcome.refused
    assert "file" in outcome.refused_reason


def test_browser_actions_are_not_yet_supported(project: Path):
    executor = Executor(sandbox=Sandbox(root=project))
    candidate = Candidate(
        key="click", description="click a thing", surface="browser",
        provenance="dom", element_ref="#id",
    )
    with pytest.raises(BrowserNotAvailable):
        executor.execute(candidate)


def test_shell_candidate_requires_argv():
    with pytest.raises(ValueError, match="no argv"):
        Candidate(key="x", description="d", surface="shell", provenance="registry")


# --------------------------------------------------------------------------
# Threshold loading
# --------------------------------------------------------------------------


def test_thresholds_load_from_the_fitted_file():
    thresholds = load_thresholds()
    assert thresholds.registry_hash is not None


def test_thresholds_fitted_against_other_wording_are_refused(tmp_path: Path):
    """Editing a criterion invalidates its threshold, and the loader must say
    so rather than quietly applying a stale cut."""
    import json

    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps(
            {"fitted": True, "registry_hash": "stale-hash", "thresholds": {"is_error": 0.5}}
        )
    )
    with pytest.raises(RuntimeError, match="re-fit|Re-fit"):
        load_thresholds(path)


def test_history_entry_records_refusals(project: Path):
    candidate = Candidate(
        key="x", description="do x", surface="shell", provenance="registry", argv=("ls",)
    )
    outcome = ActionOutcome(
        candidate=candidate, ok=False, observation="", refused_reason="denied by policy"
    )
    assert "REFUSED" in outcome.history_entry()
