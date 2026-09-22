"""Tests for the command template registry.

The registry's job is coverage without generation: it should enumerate enough
of the shell surface that ordinary inspection tasks never reach the LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools.shell import Sandbox, ShellRefused
from agent.tools.shell_registry import (
    MAX_FILE_CANDIDATES,
    REGISTRY,
    SKIP_DIRECTORIES,
    describe_state,
    enumerate_candidates,
)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# TODO: implement\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("def test_x(): pass\n")
    (tmp_path / "README.md").write_text("# Project\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    noise = tmp_path / "node_modules" / "pkg"
    noise.mkdir(parents=True)
    (noise / "index.js").write_text("module.exports = {}\n")
    return tmp_path


def test_enumerates_candidates_from_what_is_present(project: Path):
    candidates = enumerate_candidates(project)
    assert candidates
    descriptions = [c.description for c in candidates.values()]
    assert any("README.md" in d for d in descriptions)
    assert any("pyproject.toml" in d for d in descriptions)
    assert any("ls -la src" in d for d in descriptions)


def test_skips_dependency_and_build_directories(project: Path):
    """A registry that enumerated node_modules would flood the candidate list
    and, per the duplicate-description finding, collapse choice confidence."""
    descriptions = " ".join(c.description for c in enumerate_candidates(project).values())
    for skipped in SKIP_DIRECTORIES:
        assert skipped not in descriptions


def test_candidate_descriptions_are_unique(project: Path):
    """Enforced by ChoiceQuestion too, but a registry that produced duplicates
    would fail at the wire rather than here."""
    descriptions = [c.description for c in enumerate_candidates(project).values()]
    assert len(set(descriptions)) == len(descriptions)


def test_candidate_count_stays_bounded(tmp_path: Path):
    """Jev takes at most 255 options, and distinct descriptions matter more
    than volume."""
    (tmp_path / "src").mkdir()
    for i in range(200):
        (tmp_path / "src" / f"mod_{i}.py").write_text("x = 1\n")
    candidates = enumerate_candidates(tmp_path)
    cat_candidates = [c for c in candidates.values() if c.argv[0] == "cat"]
    assert len(cat_candidates) <= MAX_FILE_CANDIDATES
    assert len(candidates) < 255


def test_parameters_bind_from_task_params(project: Path):
    candidates = enumerate_candidates(project, params={"pattern": "TODO"})
    grep = next(c for c in candidates.values() if c.argv[0] == "grep")
    assert "TODO" in grep.argv
    assert grep.is_ready


def test_unbound_parameters_are_reported_not_hidden(project: Path):
    """An unbindable template is still offered: Jev selecting it is what tells
    the controller a value is needed, which is when the Value generator
    fires."""
    candidates = enumerate_candidates(project, params={})
    unready = [c for c in candidates.values() if not c.is_ready]
    assert unready
    assert all(c.unbound for c in unready)


def test_unready_candidates_can_be_excluded(project: Path):
    candidates = enumerate_candidates(project, params={}, include_unready=False)
    assert all(c.is_ready for c in candidates.values())


def test_destructive_templates_are_marked_irreversible():
    reversible = {t.name: t.reversible for t in REGISTRY}
    assert reversible["uv_sync"] is False
    assert reversible["ls"] is True
    assert reversible["git_status"] is True


def test_no_registry_template_is_denied_by_the_sandbox(tmp_path: Path):
    """A template the sandbox would refuse is a template that should not be
    offered. Catching that here beats discovering it mid-run."""
    sandbox = Sandbox(root=tmp_path)
    for template in REGISTRY:
        joined = " ".join(template.argv)
        assert sandbox.check_denied(joined) is None, f"{template.name}: {joined}"


def test_describe_state_renders_cwd_and_contents(project: Path):
    state = describe_state(project)
    assert str(project) in state
    assert "README.md" in state
    assert "No commands have been run yet." in state


def test_describe_state_includes_the_last_result(project: Path):
    state = describe_state(project, last_result="$ ls\nExit code: 0")
    assert "Exit code: 0" in state


# --------------------------------------------------------------------------
# Registry and sandbox together
# --------------------------------------------------------------------------


def test_a_selected_candidate_executes(project: Path):
    """The full shell path: enumerate, select, run."""
    sandbox = Sandbox(root=project)
    candidates = enumerate_candidates(project, params={"pattern": "TODO"})
    grep = next(c for c in candidates.values() if c.argv[0] == "grep")
    result = sandbox.run(grep.argv)
    assert result.ok
    assert "TODO" in result.stdout


def test_every_ready_candidate_survives_the_sandbox_checks(project: Path):
    """Not that every command succeeds -- git commands fail in a non-repo --
    but that none is *refused*. A refusal means the registry is offering
    something containment forbids."""
    sandbox = Sandbox(root=project)
    for candidate in enumerate_candidates(project, params={"pattern": "x"}).values():
        if not candidate.is_ready:
            continue
        try:
            sandbox.run(candidate.argv)
        except ShellRefused as exc:
            pytest.fail(f"registry offered a refused command: {candidate.argv} ({exc})")
