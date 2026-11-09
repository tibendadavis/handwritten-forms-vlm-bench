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
- Every score in Langfuse carries a comment for non-technical readers: what it means and how it was worked out, with the numbers ("right cells ÷ cells with a value … = 278 ÷ 357")
- Ten scores named for a report, the same in tables and in Langfuse (`formbench scores`): overall accuracy (the main score), completeness, reliability, rows detected, processing success rate, reporting accuracy (before and after review), error detection rate, review workload, processing time
- Truth files next to each photo (`page.json`), with columns named by key, register letter or both; unknown columns are reported, unread personal columns ignored
- `formbench eval`: one photo, one model, every wrong cell and the scores; finds the truth file next to the photo
- `formbench bench --folder`: a folder of photos with truth files, or a synthetic scenario (through review and approval), for one or more models; `--rescore` saved runs
- `formbench compare --folder <name>`: every model that ran on a folder, side by side
- Langfuse experiments: each folder is a dataset, each benchmark run a dataset run, page scores on every photo's trace, whole-run scores on the run, score definitions registered as score configs; runs locally when Langfuse is not reachable
- Prompts versioned in Langfuse prompt management: the prompt in code is saved as version 1 (`production`); `--prompt <version|label>` on `bench` and `eval`; every model call linked to its prompt version; runs named, saved and compared per prompt version; weekly runs use the `production` version
- `formbench tally`: tally sheets (totals in digit boxes, photographed one section at a time) described in a private `form.json`; one photo per section scored against its section of a whole-sheet truth file, with the same reading scores, Langfuse datasets, one versioned prompt per form and `compare`
- `formbench tally --pages`: whole-page photos of the same sheets (`<sheet>.jpg`, `<sheet>_p2.jpg`), each read in one call with its own versioned prompt (`tally-<form id>-page`) and scored on the sections printed on it (`pages` in `form.json`); `compare` shows them as "full pages"
- ADRs 0003 to 0010

### Changed
- Photos are sent byte for byte as they are in the folder: a JPEG or PNG that is upright and within `--max-side` is no longer re-compressed as JPEG (a PNG named `.jpg` used to lose quality); larger or sideways photos are still converted
- Photos are sent at up to 4032 px on the long side (was 3072), a 12 MP phone photo as taken; each run records the size it used and `compare` shows runs at another size in their own column

### Fixed
- A model reply that writes an empty cell as the text "null" or "none" is read as empty, not as an invented value; `--rescore` re-cleans saved readings

### Removed
- Tesseract OCR baseline and its CI dependency
- Generic printable forms (`make-form`), marker alignment and the per-field `extract` command, with their example schemas
- Tiles reading mode and template alignment: whole-page photos only
- `smoke` and the separate `register-*` console scripts (now `formbench` subcommands)
- Literature scores that overlapped or needed explaining: CER, WER, ANLS, ANLSL, GAnTED, Tree and field F1, STP and schema pass rates, omission and hallucination rates, cover accuracy, planted-problem recall, false alarms; and the `zss` dependency
