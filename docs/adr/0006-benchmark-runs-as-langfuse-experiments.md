# ADR 0006: Benchmark runs as Langfuse experiments, one score catalogue

- **Status:** Accepted
- **Date:** 2026-10-30

## Context

The benchmark is used to choose a model for a form. Two problems made that hard:

- **Scores were hard to read.** About 30 numbers were printed with no grouping and no
  explanation. Several overlapped (date-cell, birth-date, sex and card accuracy next to
  field accuracy).
- **Runs were hard to compare in Langfuse.** Each run was one trace with scores attached, so
  comparing two models meant opening two traces. There was no record of which photos and
  which truth a score came from.

Langfuse has datasets (items with an expected output) and experiments (a run of a task over a
dataset, with item-level and run-level scores) for exactly this. The installed SDK (3.7.0)
provides `run_experiment`, `create_dataset_item` with stable ids, score configs and run-level
scores (`create_score(dataset_run_id=...)`).

## Decision

**One score catalogue.** `evaluation/scoring.py` defines every score once: name, group,
meaning, range and which direction is better. The groups are:

- **decide:** values after review, values before review, misread flag recall, field F1, Tree F1
- **reading:** CER, WER, ANLS, ANLSL, field accuracy, GAnTED, STP, schema pass, JSON valid,
  row recall, invented rows, omission, hallucination, cover accuracy
- **review:** flags per week, planted-problem recall, false alarms, missing and extra values
- **cost:** seconds per page

The terminal tables, `formbench scores` and the Langfuse score configs all come from this
list. A new per-photo score, **values before review**, compares the daily DHIS2 values counted
from the reading with those counted from the truth. It needs no weeks or reviewer, so it also
works for a single real photo.

**Runs as experiments.**

- A folder becomes a Langfuse dataset named after the folder. The folder is either a scenario
  (`truth.json`) or real photos, each with a `<photo>.truth.json`.
- Each photo is an item with a stable id (`<folder>/<photo>`) and its truth as the expected
  output. Re-syncing updates items instead of duplicating them.
- Each `formbench bench` run of a model is one dataset run, named after the model, options
  and time.
- Every photo gets its own trace, with its page scores attached.
- Whole-run scores land on the dataset run itself: values after review, misread flag recall,
  and pooled CER and F1.
- Items run one at a time (`max_concurrency=1`), because one local model serves them.

**Langfuse stays optional.** If score configs cannot be read (no keys, or the server is
down), the same run happens on local items, and the results are still printed and saved.

## Consequences

- **Model runs can be compared in the Langfuse UI.** Under Datasets → `<folder>` → Runs,
  models sit side by side with run-level scores and per-photo drill-down.
- **Every score has an explanation in Langfuse,** and an out-of-range value is rejected.
- **The ADR 0005 baseline is unchanged.** Saved runs re-score to the new score names with
  `formbench bench --rescore`; for qwen3.5:4b the baseline numbers are identical.
- **Photo paths in Langfuse are relative to the folder,** so no local directory structure
  leaves the machine. Photos still go to the model and to the self-hosted Langfuse traces.
