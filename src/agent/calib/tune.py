"""Fit thresholds from labelled observations.

One tuner per question, because the questions have different score
distributions and different asymmetric costs. `is_error` and `irreversible`
cannot share a cut-point or a cost function.

Three things distinguish this from picking a number that looks right:

1. **Asymmetric cost.** The cut minimises `w_fn * FN + w_fp * FP`, not error
   rate. Missing a purchase is not the same as one spurious confirmation
   prompt, so they are not weighted the same.

2. **A guard band.** The chosen cut is shifted 2σ in the safe direction, σ
   being the API's own sampling jitter. A cut sitting exactly on the boundary
   flips under noise.

3. **Band population.** The fraction of observations within ±2σ of the cut is
   reported. A high band population means the question does not separate, and
   no threshold will save it -- the wording must change. This is the tuner's
   most valuable output: it can tell us a question is unusable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..jev.criteria import registry_hash
from .evaluate import Observation, load_observations

# Measured sampling jitter on repeated identical calls.
SIGMA = 0.03

# Below this many examples in the smaller class, a fitted cut says more about
# which fixtures happen to exist than about the question. The tuner reports
# such cuts as unusable for lack of data rather than as bad wording.
MIN_CLASS_SIZE = 5

THRESHOLDS_PATH = Path(__file__).resolve().parents[3] / "config" / "thresholds.json"

# Asymmetric costs. The ratio is the judgement; the absolute values are not.
COSTS: dict[str, tuple[float, float]] = {
    # (false-negative weight, false-positive weight)
    #
    # target_present: a false negative means acting confidently on a list that
    # lacks the answer -- the 0.96-confidence wrong click. A false positive
    # means an unnecessary Proposer call, which costs a fraction of a cent.
    "target_present": (10.0, 1.0),
    # is_error: a false negative wastes a run on a dead page; a false positive
    # interrupts a human for nothing.
    "is_error": (10.0, 2.0),
    # looping: same shape, slightly less costly to miss since progress
    # eventually catches it.
    "looping": (6.0, 2.0),
    # injection: gates whether the LLM reads hostile text. Missing one is a
    # security failure; firing spuriously only suppresses a Proposer call.
    "injection": (15.0, 1.0),
    # actionable: secondary signal, roughly symmetric.
    "actionable": (3.0, 3.0),
    # irreversible: missing a purchase is unacceptable. Precision can be poor;
    # an extra confirmation prompt costs a human two seconds.
    "irreversible": (50.0, 1.0),
}


@dataclass
class Fit:
    question: str
    threshold: float
    raw_threshold: float
    n_pos: int
    n_neg: int
    tpr: float
    fpr: float
    precision: float
    band_population: float
    mean_spread: float
    separation: float
    usable: bool
    note: str

    def summary(self) -> str:
        flag = "  " if self.usable else "!!"
        return (
            f"{flag} {self.question:<16} cut={self.threshold:.3f}  "
            f"TPR={self.tpr:.2f} FPR={self.fpr:.2f}  "
            f"sep={self.separation:+.2f}  band={self.band_population:.0%}  "
            f"spread={self.mean_spread:.2f}  n={self.n_pos}+/{self.n_neg}-"
        )


def _candidate_cuts(scores: list[float]) -> list[float]:
    """Cut-points to consider: the midpoints between adjacent observed values.

    Sweeping a fixed grid instead puts cuts at arbitrary positions that can
    land exactly on an observed cluster, which then makes the guard-band shift
    push the cut into the opposite class's neighbourhood and inflates band
    population. Midpoints always sit in the empty space between clusters,
    which is where a cut belongs.
    """
    unique = sorted(set(scores))
    cuts = [max(0.0, unique[0] - 0.01)]
    cuts += [(a + b) / 2 for a, b in zip(unique, unique[1:])]
    cuts.append(min(1.0, unique[-1] + 0.01))
    return cuts


def _sweep(
    scored: list[tuple[float, bool]], w_fn: float, w_fp: float
) -> tuple[float, float, float, float]:
    """Find the cut minimising asymmetric cost. Returns (cut, tpr, fpr, precision)."""
    positives = [s for s, label in scored if label]
    negatives = [s for s, label in scored if not label]
    if not positives or not negatives:
        return 0.5, 0.0, 0.0, 0.0

    best_cut, best_cost = 0.5, float("inf")
    for cut in _candidate_cuts([s for s, _ in scored]):
        fn = sum(1 for s in positives if s < cut)
        fp = sum(1 for s in negatives if s >= cut)
        cost = w_fn * fn + w_fp * fp
        if cost < best_cost:
            best_cost, best_cut = cost, cut

    tp = sum(1 for s in positives if s >= best_cut)
    fp = sum(1 for s in negatives if s >= best_cut)
    tpr = tp / len(positives)
    fpr = fp / len(negatives)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return best_cut, tpr, fpr, precision


def fit_question(
    question: str,
    scored: list[tuple[float, bool]],
    mean_spread: float = 0.0,
) -> Fit:
    w_fn, w_fp = COSTS.get(question, (1.0, 1.0))
    positives = [s for s, label in scored if label]
    negatives = [s for s, label in scored if not label]

    if not positives or not negatives:
        return Fit(
            question=question,
            threshold=0.5,
            raw_threshold=0.5,
            n_pos=len(positives),
            n_neg=len(negatives),
            tpr=0.0,
            fpr=0.0,
            precision=0.0,
            band_population=1.0,
            mean_spread=mean_spread,
            separation=0.0,
            usable=False,
            note="only one class present; cannot fit a cut",
        )

    raw_cut, tpr, fpr, precision = _sweep(scored, w_fn, w_fp)

    # Shift toward whichever side is costlier to get wrong. For a
    # false-negative-averse question that means lowering the cut, so borderline
    # cases fire rather than slip through.
    #
    # The shift is clamped so it cannot cross into the opposite class: moving
    # 2σ toward the negatives is prudent when there is room, and actively
    # harmful when the nearest negative is 2σ away. Bound it to a third of the
    # gap between the classes at the raw cut.
    direction = -1.0 if w_fn >= w_fp else 1.0
    if direction < 0:
        below = [s for s in negatives if s < raw_cut]
        room = raw_cut - max(below) if below else raw_cut
    else:
        above = [s for s in positives if s >= raw_cut]
        room = min(above) - raw_cut if above else 1.0 - raw_cut
    shift = min(2 * SIGMA, max(0.0, room / 3))
    cut = min(1.0, max(0.0, raw_cut + direction * shift))

    in_band = sum(1 for s, _ in scored if abs(s - cut) <= 2 * SIGMA)
    band_population = in_band / len(scored)
    separation = (sum(positives) / len(positives)) - (sum(negatives) / len(negatives))

    notes = []
    usable = True
    smaller_class = min(len(positives), len(negatives))

    # Distinguish "too little data to judge" from "the question genuinely
    # fails to separate". They call for opposite responses -- collect more
    # fixtures, versus reword the criterion -- and conflating them sends
    # anyone reading this report off in the wrong direction. With a small
    # corpus a single observation can be a large share of the band, so band
    # population is not interpretable until the classes are populated.
    if smaller_class < MIN_CLASS_SIZE:
        usable = False
        notes.append(
            f"only {smaller_class} example(s) in the smaller class "
            f"(want {MIN_CLASS_SIZE}+): too little data to judge this cut. "
            "Collect more fixtures of that class; do not reword yet"
        )
    elif band_population > 0.10:
        usable = False
        notes.append(
            f"{band_population:.0%} of observations sit within 2σ of the cut "
            "on a populated corpus; the question does not separate and its "
            "wording must change"
        )

    if abs(separation) < 0.15:
        usable = False
        notes.append(
            f"class means differ by only {separation:+.2f}; this question "
            "carries almost no signal regardless of where the cut goes"
        )

    if mean_spread > 0.15:
        notes.append(
            f"wording variants disagree by {mean_spread:.2f} on average; "
            "consider replacing the outlying variant"
        )

    return Fit(
        question=question,
        threshold=round(cut, 3),
        raw_threshold=round(raw_cut, 3),
        n_pos=len(positives),
        n_neg=len(negatives),
        tpr=tpr,
        fpr=fpr,
        precision=precision,
        band_population=band_population,
        mean_spread=mean_spread,
        separation=separation,
        usable=usable,
        note="; ".join(notes) if notes else "separates cleanly",
    )


def fit_all(observations: list[Observation]) -> dict[str, Fit]:
    fits: dict[str, Fit] = {}

    # Gate questions: one labelled example per fixture that labels them.
    question_names = sorted({name for o in observations for name in o.gate_truth})
    for name in question_names:
        scored = [
            (o.gate[name], o.gate_truth[name])
            for o in observations
            if name in o.gate_truth and name in o.gate
        ]
        spreads = [o.gate_spread.get(name, 0.0) for o in observations]
        mean_spread = sum(spreads) / len(spreads) if spreads else 0.0
        fits[name] = fit_question(name, scored, mean_spread)

    # Irreversibility: one labelled example per candidate, across all fixtures.
    irreversible_scored = [
        (score, o.irreversible_truth[key])
        for o in observations
        for key, score in o.irreversible.items()
        if key in o.irreversible_truth
    ]
    if irreversible_scored:
        fits["irreversible"] = fit_question("irreversible", irreversible_scored)

    # Completion: fitted on the progress score, not a noul.
    complete_scored = [
        (o.progress / 2.0, o.complete_truth)
        for o in observations
        if o.complete_truth is not None
    ]
    if complete_scored:
        fit = fit_question("complete", complete_scored)
        # Report on the 0-2 scale the policy actually uses.
        fit.threshold = round(fit.threshold * 2.0, 3)
        fit.raw_threshold = round(fit.raw_threshold * 2.0, 3)
        fits["complete"] = fit

    return fits


def write_thresholds(fits: dict[str, Fit], served_model: str) -> Path:
    payload = {
        "fitted": True,
        "registry_hash": registry_hash(),
        "jev_model": served_model,
        "sigma": SIGMA,
        "thresholds": {name: fit.threshold for name, fit in fits.items()},
        "diagnostics": {
            name: {
                "raw_threshold": fit.raw_threshold,
                "tpr": round(fit.tpr, 3),
                "fpr": round(fit.fpr, 3),
                "precision": round(fit.precision, 3),
                "separation": round(fit.separation, 3),
                "band_population": round(fit.band_population, 3),
                "mean_spread": round(fit.mean_spread, 3),
                "n_pos": fit.n_pos,
                "n_neg": fit.n_neg,
                "usable": fit.usable,
                "note": fit.note,
            }
            for name, fit in fits.items()
        },
    }
    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    return THRESHOLDS_PATH


def main() -> None:
    observations = load_observations()
    fits = fit_all(observations)
    served = sorted({o.served_model for o in observations})

    print(f"fitting thresholds from {len(observations)} fixtures ({', '.join(served)})")
    print(f"registry hash {registry_hash()}, sigma {SIGMA}\n")

    for fit in fits.values():
        print(fit.summary())
        if not fit.usable or "consider" in fit.note or "wide confidence" in fit.note:
            print(f"                   -> {fit.note}")

    starved = [
        f.question
        for f in fits.values()
        if not f.usable and min(f.n_pos, f.n_neg) < MIN_CLASS_SIZE
    ]
    broken = [f.question for f in fits.values() if not f.usable and f.question not in starved]

    print()
    if starved:
        print(f"NEEDS MORE FIXTURES: {', '.join(starved)}")
        print(
            f"  Fewer than {MIN_CLASS_SIZE} examples in the smaller class. These "
            "cuts are placeholders,\n  not judgements about the wording. Author "
            "more fixtures of the underweight class."
        )
    if broken:
        print(f"DOES NOT SEPARATE: {', '.join(broken)}")
        print(
            "  Enough data, but the question carries too little signal. Reword "
            "the criteria;\n  no cut-point will rescue these."
        )
    if not starved and not broken:
        print("all questions separate cleanly on a populated corpus")

    path = write_thresholds(fits, served[0] if served else "unknown")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
