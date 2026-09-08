# debug_client → moved

The agent's debug client now lives in its own repository and runs as the QA
sandbox UI on Cloud Run behind Cloud IAP:

**https://github.com/Anomaly-Technology/gub-sandbox-ui** (history preserved —
`git subtree split -P debug_client`, 2026-09-08).

Why it moved: as a deployed service for QA it needs the standard per-repo
pipeline (CI, WIF deployer, hooks, dependabot), which does not belong inside
the agent repo whose deploy is `adk deploy agent_engine`.

One seam still crosses the two repos: `gub-sandbox-ui/src/lib/sandbox.ts`
mirrors `gub_agent/sandbox.py` (thinking levels, defaults, the 64 KB prompt
cap, the validation rules and their wording, the variant names). A change to
the contract here is a change there — same PR train, both repos.
