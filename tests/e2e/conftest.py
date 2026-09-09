"""
End-to-end fixtures — the DEPLOYED sandbox engine, real GUB data, real models.

Opt-in, never in CI or the hooks: the whole directory is skipped unless
`SANDBOX_E2E=1`. Each test costs real tokens (~$0.05–0.15 per model) and a
minute or two of wall clock. Run it after a sandbox redeploy or an allowlist
change:

    SANDBOX_E2E=1 .venv/bin/pytest tests/e2e -q

What it needs on the operator's machine:
  - ADC (`gcloud auth application-default login`) with roles/aiplatform.user
    on the project, plus roles/iam.serviceAccountTokenCreator on the sandbox
    subject SA (terraform `sandbox_operators`);
  - the engine id (SANDBOX_AGENT_ENGINE_ID, default = the sandbox engine).

The GUB identity every run uses is the shared sandbox subject
(SANDBOX_JWT_SUBJECT_SA) — the same identity the QA UI uses — so a test
answer is what a QA tester would see. Its GUB session is revoked at the end.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field

import httpx
import pytest

PROJECT = os.environ.get("GCP_PROJECT_ID", "os-test-491819")
REGION = os.environ.get("GCP_REGION", "us-central1")
ENGINE_ID = os.environ.get("SANDBOX_AGENT_ENGINE_ID", "9148206673200939008")
PROD_ENGINE_ID = "9136379226620952576"
# The GUB the ENGINE's tools call (dev Cloud Run) — the JWT must be issued by
# it. Deliberately NOT GUB_BASE_URL: gub_agent.config loads .env, where that
# variable points at a developer's local GUB, and a local GUB refuses to be
# reached from here (connection refused) and would mint the wrong issuer anyway.
GUB_BASE_URL = os.environ.get(
    "SANDBOX_E2E_GUB_URL", "https://gcp-universal-backend-dev-843516467880.us-central1.run.app"
)
SUBJECT_SA = os.environ.get(
    "SANDBOX_JWT_SUBJECT_SA", "sa-gub-sandbox@os-test-491819.iam.gserviceaccount.com"
)
ENABLED = os.environ.get("SANDBOX_E2E") == "1"


def pytest_collection_modifyitems(config, items):  # noqa: ARG001 — pytest hook signature
    if ENABLED:
        return
    skip = pytest.mark.skip(reason="live e2e — set SANDBOX_E2E=1 to run against the sandbox engine")
    for item in items:
        if "tests/e2e" in str(item.fspath) or "tests\\e2e" in str(item.fspath):
            item.add_marker(skip)


# ── Auth ──────────────────────────────────────────────────────────────────────


def _adc_token() -> str:
    """The operator's ADC access token, via gcloud (what adk deploy uses too)."""
    out = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def _impersonated_userinfo_token(sa: str) -> str:
    """An access token AS the sandbox subject, scoped to userinfo.email — what
    GUB's access-token-exchange needs to find the user's email."""
    import google.auth
    from google.auth import impersonated_credentials
    from google.auth.transport.requests import Request

    source, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds = impersonated_credentials.Credentials(
        source_credentials=source,
        target_principal=sa,
        target_scopes=["https://www.googleapis.com/auth/userinfo.email"],
        lifetime=900,
    )
    creds.refresh(Request())
    assert creds.token, f"could not impersonate {sa}"
    return creds.token


@dataclass
class GubSession:
    jwt: str
    refresh_token: str | None
    email: str | None


@pytest.fixture(scope="session")
def gub() -> Iterator[GubSession]:
    """A GUB JWT for the sandbox subject; revoked at the end of the session."""
    token = _impersonated_userinfo_token(SUBJECT_SA)
    r = httpx.post(
        f"{GUB_BASE_URL}/auth/google/access-token-exchange",
        json={"accessToken": token},
        timeout=30,
    )
    assert r.status_code == 200, f"GUB exchange failed: {r.status_code} {r.text[:300]}"
    data = r.json()
    sess = GubSession(
        jwt=data["accessToken"],
        refresh_token=data.get("refreshToken"),
        email=(data.get("user") or {}).get("email"),
    )
    yield sess
    if sess.refresh_token:
        httpx.post(
            f"{GUB_BASE_URL}/auth/logout", json={"refreshToken": sess.refresh_token}, timeout=15
        )


# ── Engine ────────────────────────────────────────────────────────────────────


@dataclass
class Engine:
    project: str
    region: str
    engine_id: str
    token: str
    env: dict[str, str] = field(default_factory=dict)

    @property
    def base(self) -> str:
        return (
            f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project}"
            f"/locations/{self.region}/reasoningEngines/{self.engine_id}"
        )

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    @property
    def allowlist(self) -> list[str]:
        return [m for m in self.env.get("SANDBOX_MODEL_ALLOWLIST", "").split(",") if m.strip()]

    @property
    def thinking_level_models(self) -> set[str]:
        return {
            m for m in self.env.get("SANDBOX_THINKING_LEVEL_MODELS", "").split(",") if m.strip()
        }

    def create_session(self, user_id: str, state: dict) -> str:
        r = httpx.post(
            f"{self.base}:query",
            headers=self.headers,
            json={"class_method": "create_session", "input": {"user_id": user_id, "state": state}},
            timeout=60,
        )
        assert r.status_code == 200, f"create_session {r.status_code}: {r.text[:300]}"
        out = r.json().get("output", {})
        session_id = out.get("id") or out.get("session_id")
        assert session_id, f"create_session returned no id: {r.text[:300]}"
        return str(session_id)

    def stream_query(self, user_id: str, session_id: str, message: str) -> list[dict]:
        """All NDJSON events of one turn (the engine answers 200 even when the
        run died inside — an empty list IS the failure signal, see README)."""
        events: list[dict] = []
        with httpx.stream(
            "POST",
            f"{self.base}:streamQuery",
            headers=self.headers,
            json={
                "class_method": "stream_query",
                "input": {"user_id": user_id, "session_id": session_id, "message": message},
            },
            timeout=httpx.Timeout(300.0, connect=30.0),
        ) as r:
            assert r.status_code == 200, f"streamQuery {r.status_code}"
            for line in r.iter_lines():
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events


@pytest.fixture(scope="session")
def engine() -> Engine:
    """The sandbox engine + its baked env (the allowlist under test comes from
    there, never from this file — the test follows the deploy)."""
    assert ENGINE_ID != PROD_ENGINE_ID, "refusing to run e2e against the production engine"
    eng = Engine(project=PROJECT, region=REGION, engine_id=ENGINE_ID, token=_adc_token())
    r = httpx.get(eng.base, headers=eng.headers, timeout=30)
    assert r.status_code == 200, f"describe engine {r.status_code}: {r.text[:300]}"
    spec = r.json().get("spec", {}).get("deploymentSpec", {})
    eng.env = {e["name"]: e.get("value", "") for e in spec.get("env", [])}
    assert eng.env.get("SANDBOX_ENABLED") in {"1", "true", "yes"}, (
        f"engine {ENGINE_ID} is not a sandbox (SANDBOX_ENABLED={eng.env.get('SANDBOX_ENABLED')!r})"
    )
    return eng


# ── Event helpers ─────────────────────────────────────────────────────────────


def executor_text(events: list[dict], agent_name: str = "gub_agent") -> str:
    """Visible (non-thought) text the executor produced, all iterations."""
    out: list[str] = []
    for ev in events:
        if ev.get("author") != agent_name:
            continue
        for part in (ev.get("content") or {}).get("parts") or []:
            if part.get("text") and not part.get("thought"):
                out.append(part["text"])
    return "".join(out)


def sandbox_resolved(events: list[dict]) -> dict | None:
    """The `sandbox_resolved` provenance, if the echo fired (author sandbox_echo,
    actions.state_delta.sandbox_resolved)."""
    for ev in events:
        delta = ((ev.get("actions") or {}).get("state_delta") or {}).get("sandbox_resolved")
        if delta:
            return delta
    return None


def claude_reachable(project: str) -> bool:
    """Whether Anthropic models are enabled for the project (Model Garden). A
    404 'does not have access' here means every Claude arm would be an empty
    stream — the Claude cases skip instead of failing."""
    try:
        token = _adc_token()
        r = httpx.post(
            f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/anthropic/models/claude-haiku-4-5:rawPredict",
            headers={"Authorization": f"Bearer {token}", "x-goog-user-project": project},
            json={
                "anthropic_version": "vertex-2023-10-16",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "OK"}],
            },
            timeout=30,
        )
        return r.status_code == 200
    except Exception:  # noqa: BLE001 — a probe, any failure means "not reachable"
        return False
