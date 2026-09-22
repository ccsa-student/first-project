"""Command-line entry point.

    uv run agent run --task "does the test suite pass?"

Runs the shell surface only; the browser lands with the observation layer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .act.actions import Candidate
from .act.executor import Executor
from .jev.client import JevClient
from .loop.controller import Controller, ShellObserver, Supervisor, load_thresholds
from .loop.policy import StepDecision


class TerminalSupervisor(Supervisor):
    """Prompts on the terminal for confirmations and takeovers."""

    def __init__(self, auto_approve: bool = False) -> None:
        self.auto_approve = auto_approve

    def confirm(self, candidate: Candidate, reason: str) -> bool:
        print(f"\n  PAUSED: {reason}")
        print(f"  action: {candidate.description}")
        if candidate.argv:
            print(f"  command: {' '.join(candidate.argv)}")
        if self.auto_approve:
            print("  auto-approved (--yes)")
            return True
        if not sys.stdin.isatty():
            print("  no terminal to ask; refusing")
            return False
        answer = input("  run it? [y/N] ").strip().lower()
        return answer in ("y", "yes")

    def on_halt(self, decision: StepDecision, candidates: dict[str, Candidate]) -> str | None:
        print(f"\n  HALTED: {decision.reason}")
        if not sys.stdin.isatty():
            return None
        print("  available actions:")
        keys = list(candidates)
        for i, key in enumerate(keys):
            print(f"    [{i}] {candidates[key].description}")
        answer = input("  take over with which action? [number, or blank to stop] ").strip()
        if answer.isdigit() and int(answer) < len(keys):
            return keys[int(answer)]
        return None


def main() -> int:
    parser = argparse.ArgumentParser(prog="agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a task")
    run.add_argument("--task", required=True)
    run.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="bind a task parameter, e.g. --param pattern=TODO",
    )
    run.add_argument("--root", default=".", help="sandbox root (default: cwd)")
    run.add_argument("--max-steps", type=int, default=10)
    run.add_argument("--yes", action="store_true", help="auto-approve irreversible actions")
    run.add_argument(
        "--allow-provisional",
        action="store_true",
        help="run on unfitted thresholds (refused by default)",
    )
    run.add_argument("--trace", help="write the run trace to this path")

    args = parser.parse_args()

    if args.command == "run":
        params = dict(p.split("=", 1) for p in args.param if "=" in p)
        root = Path(args.root).resolve()

        thresholds = load_thresholds()
        if not args.allow_provisional:
            try:
                thresholds.require_fitted()
            except RuntimeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                print(
                    "       or pass --allow-provisional to run anyway",
                    file=sys.stderr,
                )
                return 2

        controller = Controller(
            client=JevClient.live(),
            executor=Executor(root=root),
            observer=ShellObserver(root=root, params=params),
            thresholds=thresholds,
            supervisor=TerminalSupervisor(auto_approve=args.yes),
            max_steps=args.max_steps,
            llm_available=False,
        )

        print(f"task: {args.task}")
        print(f"root: {root}\n")

        trace = controller.run(args.task)

        for step in trace.steps:
            marker = {
                "act": "->",
                "halt": "!!",
                "succeed": "OK",
                "confirm": "??",
                "reobserve": "~~",
            }.get(step.verdict, "  ")
            print(f"{marker} step {step.index}: {step.verdict}")
            print(f"     {step.reason}")
            if step.choice_description:
                print(f"     chose: {step.choice_description}")
            if step.exit_code is not None:
                print(f"     exit {step.exit_code}")
            gate = "  ".join(f"{k}={v:.2f}" for k, v in sorted(step.gate.items()))
            if gate:
                print(f"     gate: {gate}")
            print()

        print(trace.summary())

        if args.trace:
            path = trace.save(Path(args.trace))
            print(f"trace written to {path}")

        return 0 if trace.outcome in ("succeed", "max_steps") else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
