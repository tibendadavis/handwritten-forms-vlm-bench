# ADR 0007: Fewer scores with plain names; truth files by column letter

- **Status:** Accepted
- **Date:** 2026-10-31
- **Replaces:** the score list in ADR 0005 and ADR 0006

## Context

The benchmark is used to compare models on a handful of real photos with hand-written truth
files. With 25 scores, named after the literature (CER, ANLS, GAnTED, Tree F1...), it was hard
to see which model was better at the one thing that matters at this stage: putting the right
value under the right column. Truth files written by hand also tend to use the register's
column letters, which the benchmark did not accept.

## Decision

**Ten scores, each with a plain label.** A cell is one value in one row and column; it is
right when the model put the same value under the same column (dates compared as dates).

- **How well it reads:** cells correct (the main score), written cells read right, model's
  values that are right, rows found, valid output
- **What DHIS2 would get:** DHIS2 values right; for synthetic scenarios also DHIS2 values right
  after review, misreads flagged, flags per week
- **Cost:** seconds per page

Removed: CER, WER, ANLS, ANLSL, GAnTED, Tree F1, field F1, STP rate, schema pass rate, omission
and hallucination rates, invented rows, cover accuracy, planted-problem recall, false alarms,
missing and extra values. "Written cells read right" and "model's values that are right" are
the recall and precision behind field F1, which says the same thing more plainly.

**Saved runs stay comparable.** Scores are still computed from the same summed counts, so
`formbench compare` shows old runs under the new names. For the ADR 0005 baseline (qwen3.5:4b,
27 synthetic photos): cells correct 51.0%, DHIS2 values right after review 29.6%, misreads
flagged 52%, 101.1 s per page.

**Truth files by column letter.** A truth file sits next to its photo (`page.json` or
`page.truth.json`). Columns may be named by key (`bcg`), letter (`I`) or both (`I_bcg`);
column C must be split into `birth_date` and `age`. Columns the model does not read (names,
guardian, phone, village, first visit) are ignored. An unknown column stops the run with the
list of valid ones, so a typo cannot silently lower a score. The cohort month and year are
scored only when the truth has them.

**Commands.** `formbench bench --folder` (`--scenario` still works) and `formbench compare
--folder <name>`, which shows every model that ran on that folder side by side.

## Consequences

- One table answers "which model reads this form best": the first row.
- Langfuse gets new score names; old score definitions stay on the server unused.
- The removed text scores can be restored from git history if a later phase needs them.

## Update 2026-11-01: labels for reports

The labels shown in tables were renamed so results can go into a report for non-technical
readers. The score names (and so saved runs and Langfuse scores) are unchanged.

| Score name | Label |
| --- | --- |
| `cells_correct` | Overall accuracy |
| `written_cells_found` | Completeness |
| `output_values_right` | Reliability |
| `rows_found` | Rows detected |
| `valid_output` | Processing success rate |
| `dhis2_values_right` | Reporting accuracy |
| `dhis2_values_right_after_review` | Reporting accuracy after review |
| `misreads_flagged` | Error detection rate |
| `flags_per_week` | Review workload (flags per week) |
| `seconds_per_page` | Processing time (seconds per page) |

Table groups: Reading accuracy, DHIS2 reporting, Performance.
