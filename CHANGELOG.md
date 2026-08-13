# Changelog

All notable changes to this repository are tracked here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Security entries
describe what controls were added; operational detail that could narrow
an attacker's search space (which values lived where, timing of rotations,
remaining exposures) lives in internal notes instead of this public log.

## [Unreleased]

### Performance (2026-08-12)

- **Agent latency cut ~30%** (mean 44s → 31s, worst case 79s → 46s; prompt
  tokens 3.05M → 1.16M) via three measured levers, benchmarked on a prod-DB copy:
  - **Round-level cutoff** (`gub_agent/agents/round_limiter.py`, new) — caps
    ReAct at `MAX_ROUNDS` gather rounds, then strips the tool declarations so the
    model must synthesize instead of fanning out endlessly.
  - **Thinking budget MEDIUM** (`agent.py`) — the executor's unbounded dynamic
    thinking was the top latency driver (thinking_tokens ↔ elapsed r=0.86);
    MEDIUM cuts the runaway tail. Critic already runs LOW.
  - **`_sources` observation masking** (`gub_agent/agents/context_pruning.py`) —
    strips Drive citation plumbing (128k tokens on one account overview, re-sent
    every round) that the prompts already tell the model to ignore. Copy-on-write,
    so the session-shared payload is never mutated in place.
- **Circuit breaker** (`gub_agent/agents/circuit_breaker.py`, new) — per-executor-
  pass tool-call budget + identical-call dedup; a reliability guard against
  runaway fan-out / loops. Both the round and tool budgets reset at the start of
  each executor pass (`before_agent_callback`), so a critic-requested retry
  starts with a fresh budget rather than inheriting the previous pass's count.

### Added (2026-06-15)

- **Multi-agent pipeline** (`gub_agent/agents/critic.py`, `agent.py`). The
  agent is now `LoopAgent([executor, critic, escalator], max_iterations=2)` —
  a minimal critic-before-commit borrowing from the Agentic-RAG pattern. The
  critic emits a two-axis `CriticVerdict` (`info_sufficient` = did we retrieve
  enough; `answer_satisfies` = right kind of closure, grounded, recent),
  gating a retry. Same engine id; callers unaffected.
- **`org_query` tool** (`gub_agent/tools/org_query.py`) — structured
  query primitive (filter/sort/group/aggregate/`similar_to`) preferred for
  all filter/count/sort/aggregate questions.
- **Deterministic current-date injection** (`gub_agent/instruction_utils.py`).
  Executor + critic use an ADK `InstructionProvider` (callable instruction)
  that appends today's UTC date per request — no LLM-mediated tool, never
  stale — so recency reasoning ("this week") is reliable.
- **Synthesis principles** in the executor prompt: author to the question's
  intended *closure* (fact → value, assessment → verdict, exploratory →
  shortlist), ground every named entity, weight recency. General, not tuned
  to specific phrasings.
- **`debug_client/`** — local-only Next.js tool for inspecting the agent's
  decomposition trace (tool calls, two-axis critic verdict, sources).

### Added (2026-04-23)

- `README.md` — project overview, local setup, deploy pointers. The
  repo had no README before.

### Security (2026-04-23)

- Added a `gitleaks` pre-commit hook (`.githooks/pre-commit`) that scans
  staged changes before every commit and rejects the commit on any
  finding. One-time setup per clone is documented in the README
  ("Local setup").
- Tightened `.gitignore` coverage to block additional secret-file shapes
  from slipping through (additional `.env.*` variants, typical key/cert
  file extensions, common service-account JSON name patterns).
