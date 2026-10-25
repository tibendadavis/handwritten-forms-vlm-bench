# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Project scaffold (`formbench` package, uv-managed, pinned dependencies)
- Pre-commit hooks: gitleaks secret scanning, ruff lint and format
- GitHub Actions CI: lint, tests, secrets scan on every push
- Smoke test: single traced VLM call via Ollama's OpenAI-compatible API, recorded in self-hosted Langfuse

### Fixed
- `.env.example` uses `LANGFUSE_HOST`, the variable the installed Langfuse SDK reads
