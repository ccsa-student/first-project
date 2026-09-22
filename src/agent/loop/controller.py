"""The agent loop.

    observe -> assess (Call A) -> decide -> guard (Call B) -> execute -> repeat

The controller owns orchestration and nothing else. Every decision is made by
`policy.decide` and `policy.guard`, which are pure; the controller's job is to
gather observations, make the calls, and act on the verdict. That separation
is what keeps the decision logic testable without a network or a browser.

Confirmation and takeover are surfaced through a `Supervisor`, so the same
loop serves the CLI (which prompts on the terminal) and the dashboard (which
prompts in a browser) without knowing which it is talking to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..act.actions import ActionOutcome, Candidate
from ..act.executor import Executor
from ..jev.client import JevClient
from ..jev.criteria import registry_hash
from ..jev.panels import aggregate, build_assess_panel, build_guard_panel
from ..jev.schema import NoulAnswer
from ..tools.shell_registry import describe_state, enumerate_candidates
from .policy import StepDecision, Thresholds, decide, guard
from .trace import RunTrace, record_step

THRESHOLDS_PATH = Path(__file__).resolve().parents[3] / "config" / "thresholds.json"


def load_thresholds(path: Path = THRESHOLDS_PATH) -> Thresholds:
    """Load fitted thresholds, refusing ones fitted against other wording."""
    if not path.exists():
        return Thresholds()

    payload = json.loads(path.read_text())
    values = payload.get("thresholds", {})

    if payload.get("registry_hash") != registry_hash():
        raise RuntimeError(
            f"thresholds in {path} were fitted against criteria wording "
            f"{payload.get('registry_hash')}, but the registry now hashes to "
            f"{registry_hash()}. Re-fit: uv run python -m agent.calib.tune"
        )

    known = set(Thresholds.__dataclass_fields__) - {
        "fitted",
        "registry_hash",
        "jev_model",
    }
    return Thresholds(
        **{k: v for k, v in values.items() if k in known},
        fitted=payload.get("fitted", False),
        registry_hash=payload.get("registry_hash"),
        jev_model=payload.get("jev_model"),
    )


class Supervisor(Protocol):
    """How the loop reaches a human."""

    def confirm(self, candidate: Candidate, reason: str) -> bool:
        """Approve an irreversible action. False aborts it."""

    def on_halt(self, decision: StepDecision, candidates: dict[str, Candidate]) -> str | None:
        """A halt happened. Return a candidate key to take over with, or None
        to end the run."""


class AutoDeny(Supervisor):
    """Non-interactive default: never approve, never take over.

    Deliberately conservative -- an unattended run should stop at an
    irreversible action, not guess.
    """

    def confirm(self, candidate: Candidate, reason: str) -> bool:
        return False

    def on_halt(self, decision: StepDecision, candidates: dict[str, Candidate]) -> str | None:
        return None


@dataclass
class ShellObserver:
    """Produces the observation and candidate list for the shell surface."""

    root: Path
    params: dict[str, str]
    last_result: str | None = None

    def observe(self) -> tuple[str, dict[str, Candidate]]:
        observation = describe_state(self.root, self.last_result)
        candidates = {
            key: Candidate(
                key=key,
                description=shell_candidate.description,
                surface="shell",
                provenance="registry",
                argv=shell_candidate.argv,
                reversible=shell_candidate.reversible,
                unbound=shell_candidate.unbound,
            )
            for key, shell_candidate in enumerate_candidates(self.root, self.params).items()
        }
        return observation, candidates


def compose_state(
    task: str,
    params: dict[str, str],
    history: list[str],
    observation: str,
    candidates: dict[str, Candidate],
    surface_heading: str = "SHELL",
) -> str:
    """Render the Jev state string.

    The AVAILABLE ACTIONS section is not decoration: target_present asks
    whether the needed action appears in it, so omitting it would silently
    disable the primary gate.
    """
    parts = [f"## TASK\n{task}"]

    if params:
        bound = "\n".join(f"  {k} = {v}" for k, v in sorted(params.items()))
        parts.append(f"## PARAMETERS\n{bound}")

    if history:
        recent = history[-5:]
        steps = "\n".join(f"  {i + 1}. {entry}" for i, entry in enumerate(recent))
        parts.append(f"## PROGRESS\nstep {len(history) + 1}. Recent actions:\n{steps}")
    else:
        parts.append("## PROGRESS\nstep 1. No actions taken yet.")

    parts.append(f"## {surface_heading}\n{observation}")

    actions = "\n".join(f"  - {c.description}" for c in candidates.values())
    parts.append(f"## AVAILABLE ACTIONS\n{actions}")

    return "\n".join(parts)


class Controller:
    def __init__(
        self,
        client: JevClient,
        executor: Executor,
        observer: ShellObserver,
        thresholds: Thresholds,
        supervisor: Supervisor | None = None,
        max_steps: int = 25,
        llm_available: bool = False,
    ) -> None:
        self.client = client
        self.executor = executor
        self.observer = observer
        self.thresholds = thresholds
        self.supervisor = supervisor or AutoDeny()
        self.max_steps = max_steps
        self.llm_available = llm_available

    def run(self, task: str) -> RunTrace:
        trace = RunTrace(
            task=task,
            params=dict(self.observer.params),
            thresholds_hash=self.thresholds.registry_hash or "provisional",
        )
        history: list[str] = []

        for index in range(self.max_steps):
            observation, candidates = self.observer.observe()
            if len(candidates) < 2:
                trace.outcome = "halt"
                trace.add(
                    record_step(
                        index,
                        StepDecision(
                            verdict="halt",
                            reason="fewer than two candidate actions; nothing to choose between",
                        ),
                        choice_description=None,
                        choice_confidence=0.0,
                        provenance=None,
                        candidate_count=len(candidates),
                        jev_input_tokens=0,
                        jev_calls=0,
                        served_model="",
                    )
                )
                return trace

            state = compose_state(task, self.observer.params, history, observation, candidates)
            descriptions = {k: c.description for k, c in candidates.items()}

            response = self.client.ask(state, build_assess_panel(descriptions))
            panel = aggregate(response)
            trace.jev_model = response.model

            decision = decide(
                panel,
                self.thresholds,
                llm_available=self.llm_available,
            )
            jev_calls = 1
            jev_tokens = response.usage.input_tokens
            outcome: ActionOutcome | None = None

            if decision.verdict in ("halt", "propose", "succeed"):
                # `propose` without an LLM is already converted to a halt by
                # the policy, so reaching it here means the LLM layer exists
                # and the controller has not been taught to call it yet.
                if decision.verdict == "propose":
                    raise NotImplementedError(
                        "the Proposer is not wired in; run with llm_available=False"
                    )

                if decision.verdict == "halt":
                    takeover = self.supervisor.on_halt(decision, candidates)
                    if takeover and takeover in candidates:
                        outcome = self.executor.execute(candidates[takeover])
                        history.append(f"[human] {outcome.history_entry()}")
                        self.observer.last_result = outcome.observation
                        trace.add(
                            record_step(
                                index,
                                decision,
                                choice_description=candidates[takeover].description,
                                choice_confidence=panel.choice_confidence,
                                provenance="human",
                                candidate_count=len(candidates),
                                jev_input_tokens=jev_tokens,
                                jev_calls=jev_calls,
                                served_model=response.model,
                                outcome=outcome,
                            )
                        )
                        continue

                trace.outcome = decision.verdict
                trace.add(
                    record_step(
                        index,
                        decision,
                        choice_description=None,
                        choice_confidence=panel.choice_confidence,
                        provenance=None,
                        candidate_count=len(candidates),
                        jev_input_tokens=jev_tokens,
                        jev_calls=jev_calls,
                        served_model=response.model,
                    )
                )
                return trace

            chosen = candidates[decision.action_key]  # type: ignore[index]

            # ---- Call B: the guard panel ---------------------------------
            guard_state = (
                f"{state}\n\n## ACTION UNDER REVIEW\n{chosen.description}"
            )
            guard_response = self.client.ask(guard_state, build_guard_panel())
            jev_calls += 1
            jev_tokens += guard_response.usage.input_tokens

            raw = {
                key: answer.noul
                for key, answer in guard_response.answers.items()
                if isinstance(answer, NoulAnswer)
            }
            from ..jev.criteria import GUARD_PANEL

            guard_gate = {q.name: q.aggregate(raw) for q in GUARD_PANEL}

            # A template that declares itself irreversible is confirmed
            # regardless of what the classifier returns. The declaration is an
            # author's statement of fact about the command, and the noul is a
            # judgement under uncertainty; the same reasoning that keeps the
            # deny-list independent of Jev applies here. Measured on `uv sync`,
            # whose noul sits below the fitted cut: without this the agent ran
            # a declared-irreversible command with no prompt.
            if not chosen.reversible:
                guard_gate["irreversible"] = max(
                    guard_gate.get("irreversible", 0.0), 1.0
                )

            guard_decision = guard(chosen.key, guard_gate, self.thresholds)
            if not chosen.reversible and guard_decision.verdict == "confirm":
                guard_decision = StepDecision(
                    verdict="confirm",
                    reason=(
                        "the command template declares this action irreversible, "
                        "so it is confirmed regardless of the classifier"
                    ),
                    action_key=guard_decision.action_key,
                    gate=guard_decision.gate,
                )

            if guard_decision.verdict == "reobserve":
                history.append(f"{chosen.description} -- skipped, target no longer present")
                trace.add(
                    record_step(
                        index,
                        guard_decision,
                        choice_description=chosen.description,
                        choice_confidence=panel.choice_confidence,
                        provenance=chosen.provenance,
                        candidate_count=len(candidates),
                        jev_input_tokens=jev_tokens,
                        jev_calls=jev_calls,
                        served_model=response.model,
                    )
                )
                continue

            if guard_decision.verdict == "confirm":
                approved = self.supervisor.confirm(chosen, guard_decision.reason)
                if not approved:
                    trace.outcome = "confirm_denied"
                    trace.add(
                        record_step(
                            index,
                            guard_decision,
                            choice_description=chosen.description,
                            choice_confidence=panel.choice_confidence,
                            provenance=chosen.provenance,
                            candidate_count=len(candidates),
                            jev_input_tokens=jev_tokens,
                            jev_calls=jev_calls,
                            served_model=response.model,
                        )
                    )
                    return trace

            outcome = self.executor.execute(chosen)
            history.append(outcome.history_entry())
            self.observer.last_result = outcome.observation

            trace.add(
                record_step(
                    index,
                    decision,
                    choice_description=chosen.description,
                    choice_confidence=panel.choice_confidence,
                    provenance=chosen.provenance,
                    candidate_count=len(candidates),
                    jev_input_tokens=jev_tokens,
                    jev_calls=jev_calls,
                    served_model=response.model,
                    outcome=outcome,
                )
            )

        trace.outcome = "max_steps"
        return trace
