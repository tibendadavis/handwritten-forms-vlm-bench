# ADR 0005: Text, field and structure scores for register transcription

- **Status:** Accepted
- **Date:** 2026-10-28

## Context

Benchmark scores must reflect three things the data owner asked for:

- how well text is read, letter by letter and word by word (names, dates, facility names)
- how many fields, and whole forms, come out right
- whether values stay under the correct header in a DHIS2-compatible format

Our earlier scores (date-cell accuracy, approved-value accuracy) answer the integrity question
but are not the standard document-extraction measures. Full-page reading is now the default
mode, because users photograph the whole (A3-sized) register page.

## Decision

Every page is scored with three groups of metrics. They are recorded per page, on that page's
Langfuse trace, and per run on the Langfuse dataset run ([ADR 0006](0006-benchmark-runs-as-langfuse-experiments.md)).
Implementation: `metrics.py` (generic) and `evaluation/scoring.py` (register-specific), which
also lists every score with its meaning (`formbench scores`).

**How text is compared.**

- **Whitespace** is collapsed.
- **Dates** are compared as digits only for text scores, and by calendar date for field scores.
  People write `12-06-26`, `12/6/26` or stack the date over lines in a narrow column; those are
  the same value, not reading errors.
- **Rows** are paired between truth and prediction by optimal (Hungarian) matching on content,
  so one invented row does not shift every comparison after it.

**Metrics.**

| Group | Metric | Definition |
| --- | --- | --- |
| Text | `cer`, `wer` | Character and word edit distance / truth length, corpus-level over every non-empty truth text field (dates, birth dates, ages, card numbers, serials, cohort Month/Year, cover fields). |
| Text | `anls` | Mean ANLS per field: 1 − normalized Levenshtein distance, 0 when the distance is ≥ 0.5. |
| Text | `anlsl` | ANLS for **lists** (as in ANLS*): rows matched optimally, summed similarity / max(truth rows, predicted rows), so missing and invented rows both cost. |
| Fields | `field_accuracy` | Correct cells / cells that carry a value in the truth or the prediction. Blank-on-blank is not counted, so the score is not inflated by empty cells. |
| Fields | `field_f1` | TP = non-empty and correct; FP = non-empty and wrong, or invented; FN = truth value missed or wrong. |
| Documents | `stp_rate` | Share of page photos with zero errors (straight-through processing). |
| Structure | `ganted` | 1 − ANLS-weighted tree edit distance / truth tree size (Zhang–Shasha via `zss`). Inserting or deleting a node costs 1; relabelling a leaf costs its normalized edit distance. It is computed per matched row group (unmatched rows cost their full size) to keep a page under a second; a whole-page tree takes about 14 s. |
| Structure | `tree_f1` | F1 over (row serial, field) → value paths. A correct value under the wrong column or child counts as wrong, which tests "stays under the correct header". |
| Structure | `schema_pass_rate` | Share of outputs that pass the JSON Schema **and** DHIS2 format checks: every date is a real date, sex is M/F, the cohort parses, serials are numeric. |
| Cover | `cover_field_accuracy` | Facility name, MFL code, facility type, district and region. Their text is also included in CER, WER and ANLS. |

The integrity scores from ADR 0004 stay alongside these, unchanged:

- review workload: `flags_per_week`, `edge_recall`, `misread_flag_recall`, `false_alarms`
- `value_accuracy_after_review`: approved daily values after review
- `value_accuracy_before_review` (added 2026-10-30): daily values counted from one page's
  reading vs from its truth, with no review, so a single real photo has a DHIS2-level score

## Consequences

- **The scores separate reading from placement.** High ANLS with low Tree F1 means digits are
  read but put under the wrong column, which is the main full-page failure we see.
- **Baseline, 2026-10-28:** qwen3.5:4b, full page, 6 synthetic scenarios, 27 page photos
  from neat writing on flat scans to terrible writing in terrible photos.
  - **Overall:**
    - CER 0.38, WER 0.46, ANLS 0.61, ANLSL 0.63
    - field accuracy 51%, field F1 0.58, STP rate 3.7%
    - GAnTED 0.70, Tree F1 0.56, schema pass rate 48%
    - 30% of daily values correct after review
    - about 100 s per page
  - **Range:** field F1 runs from 0.85 (neat scan) down to 0.16 (terrible).
  - **Main failure mode:** dates read correctly but placed under the neighbouring narrow
    column.
- **Interpretations to confirm.** "GAnTED" and "ANLSL" are implemented as described above
  (grouped, ANLS-weighted nTED; ANLS over lists). If the data owner's reference defines them
  differently, only `evaluation/scoring.py` changes.
- **Real photos need a truth file.** For a real photo, a page reading corrected by hand is the
  truth file for `formbench eval`, and a folder of such photos can be benchmarked like a
  scenario.
