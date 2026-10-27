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
- `smoke` console script (`uv run smoke IMAGE MODEL`); `.env` is loaded automatically via python-dotenv
- Schema generator: DHIS2 data set metadata -> extraction JSON Schema (unique keys, nullable, no extra fields) and mapping to dataElement/categoryOptionCombo UIDs; metadata snapshot for offline, repeatable regeneration
- `make-form`: printable one-page A4 form (PDF/PNG, 300 DPI) for a seeded selection of data elements, with corner alignment markers, `layout.json` cell boxes, and the subset schema and mapping for exactly the printed fields
- Seeded ground truth (`truth.json`: made-up facility, valid period for the periodType, counts with a configurable blank ratio) and a printable `values_sheet.pdf` for filling forms by hand
- Photo alignment to the form template via the four corner markers (OpenCV perspective warp)
- Tesseract template baseline: per-cell crop, empty-cell detection by ink ratio, digit-whitelisted OCR with per-field confidence
- VLM extraction through Ollama's OpenAI-compatible API with a strict JSON Schema response format, temperature 0, fixed seed, thinking off by default (`--think` to enable), output validated with jsonschema
- Field-level scoring: `json_valid`, `field_accuracy`, `numeric_accuracy`, `hallucination_rate`, `omission_rate`
- `extract` CLI: per-field result table, metric summary with latency, run files in `results/runs/`, Langfuse trace with metadata and scores
- ADRs 0002 (template Tesseract baseline) and 0003 (VLM structured output and thinking); README "Try it" walkthrough
- CI installs Tesseract so OCR tests run there

### Fixed
- `.env.example` uses `LANGFUSE_HOST`, the variable the installed Langfuse SDK reads
