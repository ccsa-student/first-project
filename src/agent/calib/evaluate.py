"""Run fixtures through Jev and collect raw observations.

Separated from metrics deliberately: this module produces records, metrics
interprets them, and the tuner fits cut-points to them. Keeping them apart
means a threshold can be re-fitted from existing cassettes without another
API call.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..jev.client import JevClient
from ..jev.criteria import GATE_PANEL, IRREVERSIBLE, registry_hash
from ..jev.panels import aggregate, build_assess_panel, spread
from .fixtures import Fixture, cassette_path, load_all


@dataclass
class Observation:
    """What Jev said about one fixture, plus what the labels say is true."""

    fixture_id: str
    fixture_class: str
    surface: str

    gate: dict[str, float]
    gate_spread: dict[str, float]
    gate_truth: dict[str, bool]

    choice: str | None
    choice_confidence: float
    probabilities: dict[str, float]

    progress: float
    progress_confidence: float
    complete_truth: bool | None

    # Per-candidate irreversibility, keyed by candidate. Labels say which
    # candidates are genuinely irreversible.
    irreversible: dict[str, float] = field(default_factory=dict)
    irreversible_truth: dict[str, bool] = field(default_factory=dict)

    correct: tuple[str, ...] = ()
    acceptable: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()

    served_model: str = ""
    input_tokens: int = 0

    @property
    def picked_correct(self) -> bool:
        if not (self.correct or self.acceptable):
            return False
        return self.choice in set(self.correct) | set(self.acceptable)

    @property
    def picked_forbidden(self) -> bool:
        return self.choice in set(self.forbidden)


def evaluate_fixture(fixture: Fixture, client: JevClient) -> Observation:
    """One assess call plus one batched irreversibility panel.

    Irreversibility is asked about every candidate rather than only the chosen
    one. Latency is flat in question count, so this costs nothing extra and
    yields recall data on candidates the agent did not pick -- which is what
    the recall metric actually needs.
    """
    response = client.ask(fixture.state(), build_assess_panel(fixture.candidates))
    panel = aggregate(response)

    irreversible_questions = {}
    for key, description in fixture.candidates.items():
        for i, (true_pole, false_pole) in enumerate(IRREVERSIBLE.variants):
            irreversible_questions[f"irr__{key}__v{i}"] = {
                "type": "noul",
                "criteria": {
                    "true": f"{true_pole}. The action under review is: {description}",
                    "false": f"{false_pole}. The action under review is: {description}",
                },
            }

    irr_response = client.ask(
        f"{fixture.state()}\n\n## ACTIONS UNDER REVIEW\nEach question names one action.",
        irreversible_questions,
    )

    irreversible: dict[str, float] = {}
    for key in fixture.candidates:
        values = [
            irr_response.noul(f"irr__{key}__v{i}")
            for i in range(len(IRREVERSIBLE.variants))
        ]
        irreversible[key] = sum(values) / len(values)

    return Observation(
        fixture_id=fixture.id,
        fixture_class=fixture.fixture_class,
        surface=fixture.surface,
        gate=panel.gate,
        gate_spread={q.name: spread(q.name, response) for q in GATE_PANEL},
        gate_truth=fixture.labels.gate_truth(),
        choice=panel.choice,
        choice_confidence=panel.choice_confidence,
        probabilities=panel.probabilities,
        progress=panel.progress,
        progress_confidence=panel.progress_confidence,
        complete_truth=fixture.labels.complete,
        irreversible=irreversible,
        irreversible_truth={
            key: key in set(fixture.labels.irreversible) for key in fixture.candidates
        },
        correct=fixture.labels.correct,
        acceptable=fixture.labels.acceptable,
        forbidden=fixture.labels.forbidden,
        served_model=panel.served_model,
        input_tokens=response.usage.input_tokens + irr_response.usage.input_tokens,
    )


def evaluate_all(mode: str = "replay") -> list[Observation]:
    """Evaluate every fixture.

    mode="replay" reads cassettes only and never touches the network.
    mode="record" hits the live API and writes cassettes.
    """
    observations = []
    for fixture in load_all():
        path = cassette_path(fixture)
        client = (
            JevClient.recording(path) if mode == "record" else JevClient.replaying(path)
        )
        observations.append(evaluate_fixture(fixture, client))
    return observations


OBSERVATIONS_PATH = Path(__file__).resolve().parents[3] / "config" / "observations.json"


def save_observations(observations: list[Observation], path: Path = OBSERVATIONS_PATH) -> None:
    payload = {
        "registry_hash": registry_hash(),
        "served_models": sorted({o.served_model for o in observations}),
        "observations": [asdict(o) for o in observations],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_observations(path: Path = OBSERVATIONS_PATH) -> list[Observation]:
    payload = json.loads(path.read_text())
    if payload["registry_hash"] != registry_hash():
        raise ValueError(
            "observations were recorded against different criteria wording "
            f"({payload['registry_hash']} vs {registry_hash()}). Re-record: "
            "uv run python -m agent.calib.evaluate --record"
        )
    return [Observation(**row) for row in payload["observations"]]


def main() -> None:
    import sys

    mode = "record" if "--record" in sys.argv else "replay"
    print(f"evaluating fixtures in {mode} mode")
    observations = evaluate_all(mode)
    save_observations(observations)
    print(f"  {len(observations)} fixtures evaluated")
    print(f"  served models: {sorted({o.served_model for o in observations})}")
    print(f"  wrote {OBSERVATIONS_PATH}")


if __name__ == "__main__":
    main()
