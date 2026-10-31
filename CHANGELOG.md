# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Project scaffold (`formbench` package, uv-managed, pinned dependencies), pre-commit hooks (gitleaks, ruff), GitHub Actions CI (lint, tests, secrets scan)
- One `formbench` command: `eval`, `bench`, `compare`, `scores`, `synth`, `template`, `metadata`, `run`, `approve`; `.env` is loaded automatically
- VLM calls through Ollama's OpenAI-compatible API with a strict JSON Schema response format, temperature 0, fixed seed, thinking off by default (`--think`), capped output and no silent retries; every call traced in Langfuse
- Read-only DHIS2 client: data set metadata download (`formbench metadata`) and org unit lookup by code
- Immunisation register: column-to-DHIS2 mapping from metadata, a page schema that cannot return names or contacts, strict parsing of handwritten dates and ages, whole-page reading
- Deterministic daily aggregation by dose date with provenance; Children vaccinated as distinct children; blocking flags instead of guesses
- Weekly workflow: draft with flags, `review.html` and `decisions.yaml`, approval by a named person, `datavalueset.json` (never posted), corrections as full totals for already approved weeks
- Synthetic registers on the real blank page: simulated children and visits over several weeks, planted edge cases, one to three writers from neat to terrible, pencil and faded ink, scan to terrible photos; same seed, same photos
- Scores, defined once with their meaning, range and direction (`formbench scores`): values correct after and before review, misread flag recall, field and Tree F1, CER, WER, ANLS, ANLSL, field accuracy, GAnTED, STP rate, schema pass rate, omission and hallucination rates, review workload, latency
- `formbench eval`: one photo, one model, every wrong cell and the scores
- `formbench bench`: a synthetic scenario (through review and approval) or a folder of photos with truth files, for one or more models; `--rescore` saved runs; `formbench compare` combines runs per model
- Langfuse experiments: each folder is a dataset, each benchmark run a dataset run, page scores on every photo's trace, whole-run scores on the run, score definitions registered as score configs; runs locally when Langfuse is not reachable
- ADRs 0003 to 0006

### Removed
- Tesseract OCR baseline and its CI dependency
- Generic printable forms (`make-form`), marker alignment and the per-field `extract` command, with their example schemas
- Tiles reading mode and template alignment: whole-page photos only
- `smoke` and the separate `register-*` console scripts (now `formbench` subcommands)
