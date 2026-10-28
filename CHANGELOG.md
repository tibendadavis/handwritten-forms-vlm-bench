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
- R-06 register pipeline: daily RI v2 values from photographed immunisation register pages, replacing weekly tally sheets
  - `register-template`: renders the blank register PDF and detects its grid and header areas
  - Strict parsing of handwritten dates and the register's age notation
  - Column-to-vaccine/dose mapping with data element and category option combo UIDs resolved from DHIS2 metadata; age bands parsed from COC names
  - Extraction schemas that cannot return names, guardians, phone numbers or villages
  - Page reading in two modes: whole photo, or aligned 4-row strips with only the counting columns (ORB + RANSAC + ECC alignment, grid-overlap check)
  - Deterministic daily aggregation by dose date with provenance; Children vaccinated as distinct children; blocking flags instead of guesses
  - Approved-reading state per register book: re-read discrepancies, late entries as full-total corrections, reviewer fixes carried forward
  - `register-run`: draft, self-contained `review.html` and `decisions.yaml`; `register-approve`: refuses open flags, requires typed confirmation, writes `datavalueset.json` (never posted)
  - `register-synth`: multi-week synthetic scenarios with injected edge cases, handwritten on the real blank page and photographed in three quality tiers
  - `register-bench`: transcription, review-workload, flag-catch and approved-value scores per model and mode, traced in Langfuse
  - ADR 0004 (register to daily aggregates)
- Read-only org unit lookup by code in the DHIS2 client
- Register evaluation scores (ADR 0005): CER, WER, ANLS, ANLSL, field accuracy, field F1, STP rate, GAnTED, Tree F1, schema pass rate and cover accuracy, per page (own Langfuse span) and per run
- `register-eval`: test one photo with one model; shows every cell read differently from the truth, scores it and saves the reading
- `register-bench --summary`: combine saved runs across scenarios per model and mode; `--max-side` for full-page resolution
- Synthetic registers: one to three writers per page, messiness levels up to terrible handwriting, pencil and faded ink, spill over cell borders, an A3-sized register and a terrible photo tier

### Changed
- Register reading defaults to the whole page (`--mode full`), since users photograph the whole register page

### Fixed
- `register-bench` failed on start because `--max-side` was read but never registered
- `.env.example` uses `LANGFUSE_HOST`, the variable the installed Langfuse SDK reads
