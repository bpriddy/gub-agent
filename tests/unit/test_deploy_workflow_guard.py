"""The Deploy workflow must refuse to run from any ref but main.

Parses the workflow YAML and checks that the first step of every job that can
deploy is an explicit main-only check, that it needs no checkout or credentials
to run, and that the ref is passed through env (never interpolated in run:).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "deploy.yml"


def _jobs() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]


def _deploys(job: dict) -> bool:
    return any("adk deploy" in (s.get("run") or "") for s in job["steps"])


def test_workflow_has_a_deploying_job():
    assert any(_deploys(j) for j in _jobs().values())


def test_guard_is_first_step_of_every_deploying_job():
    for name, job in _jobs().items():
        if not _deploys(job):
            continue
        first = job["steps"][0]
        assert "uses" not in first, f"{name}: first step must be the guard, not an action"
        run = first.get("run", "")
        assert first["env"]["DEPLOY_REF"] == "${{ github.ref }}", name
        for ref, code in [
            ("refs/heads/main", 0),
            ("refs/heads/feature", 1),
            ("refs/heads/main2", 1),
            ("refs/tags/x", 1),
            ("", 1),
        ]:
            r = subprocess.run(
                ["bash", "-c", run], env={"DEPLOY_REF": ref, "PATH": "/usr/bin:/bin"}
            )
            assert r.returncode == code, f"{name}: ref={ref!r}"


def test_no_expression_interpolated_in_guard_script():
    for job in _jobs().values():
        assert "${{" not in job["steps"][0].get("run", "")


def test_no_job_needs_gate_bypass():
    for name, job in _jobs().items():
        assert "if" not in job, f"{name}: job-level if could skip the guard"
        assert "if" not in job["steps"][0], f"{name}: guard must be unconditional"


def test_guard_cannot_be_neutralised():
    wf = yaml.safe_load(WORKFLOW.read_text())
    assert set(wf.get(True, wf.get("on"))) == {"workflow_dispatch"}
    for name, job in wf["jobs"].items():
        for key in ("if", "needs", "continue-on-error"):
            assert key not in job, f"{name}: job-level {key}"
        for key in ("if", "continue-on-error"):
            assert key not in job["steps"][0], f"{name}: guard step {key}"
