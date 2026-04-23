# Changelog

All notable changes to this repository are tracked here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Security entries
describe what controls were added; operational detail that could narrow
an attacker's search space (which values lived where, timing of rotations,
remaining exposures) lives in internal notes instead of this public log.

## [Unreleased]

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
