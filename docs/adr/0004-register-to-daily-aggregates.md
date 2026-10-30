# ADR 0004: Daily RI v2 values from the R-06 register, with human approval

- **Status:** Accepted
- **Date:** 2026-10-28

## Context

Health workers record every immunisation in the R-06 Immunisation Register. A page is a birth
cohort, with one row per child and a date written under each vaccine given. Every week they
then re-count the same doses onto F4A tally sheets and enter those totals into the DHIS2
**RI v2** data set, which has a **Daily** period type. That second step costs time and adds
transcription errors. The goal is to remove it while holding data integrity to the highest
standard.

What the register implies:

- **A page is not a period.** The page Month/Year is the children's *birth* month. Doses
  given on one day are spread over many cohort pages, and a page keeps receiving doses for
  about two years.
- **Every dose has its own date.** That is enough to compute daily values without tallying.
- **The cover holds the facility details:** Facility Name, MFL Code and Facility Type.
- **Some RI v2 values are not in the register at all:** inventory, missed opportunities
  and Unique Number of Forms.

## Decision

**Two stages: the model transcribes, code counts.** The VLM only copies what is written in
the counting columns (serial, card number, sex, birth date, age, dose dates). Deterministic,
tested code turns that into RI v2 values. The model never sees or produces a count or a UID.

**Rules, agreed with the data owner:**

- **Counting by date.**
  - A filled date means the dose was given that day, and it counts on that day: one daily
    period, delivered weekly per ISO 8601 week.
  - The page's Month/Year never decides the period.
  - Doses dated before the global pilot go-live date (`REGISTER_GO_LIVE`) are listed but
    never counted. They were already reported through tally sheets.
- **Mapping register columns to data elements.**
  - BCG and OPV 0 → Dose 0.
  - OPV, Pentavalent, IPV and PCV → Doses 1–3 (IPV 1–2).
  - Rota → Dose 1.
  - MCV 1/2 → MRI Dose 1/2.
  - Column J (Polio) is read but not counted until its mapping is decided.
- **Disaggregation follows the metadata.**
  - The age band is the child's age in completed months on the dose date, from the birth date.
  - Bands and category option combos are parsed from the data set metadata, which is the
    source of truth. An age with no band is not counted.
- **Children vaccinated** counts distinct children (rows) with at least one counted dose that
  day, per sex × age band. It is never a sum of doses.
- **Problems are flagged, never guessed.** These block the affected row or cell until a
  reviewer decides:
  - missing or unreadable birth date or sex
  - a row whose birth month is not the page's cohort
  - unreadable, future or pre-birth dates
  - duplicate serials
  - a blank or unmatched Facility Type, which supplies the session-type attribute option
    combo; "Not filled" and "Unclear" are never chosen automatically
- **Pages are photographed again later.**
  - Approved page readings are kept locally per register book.
  - A cell that reads differently from its approved reading is flagged.
  - A dose that appears late for an already approved week becomes a **correction** with the
    new **full** total for that day, because a DHIS2 import overwrites the stored value.
  - Reviewer fixes carry over to later re-reads of the same row.
- **Human in the loop, always.**
  - A named person approves every week, even when there are no flags.
  - Only then is the `dataValueSet` JSON written. It is never pushed to DHIS2.
  - The org unit is matched from the MFL Code by a read-only lookup. If no unit matches, the
    run warns and `orgUnit` is null.
- **Privacy.**
  - The model receives the whole page photo (tiles mode, which cut out personal columns,
    was removed on 2026-10-30; see below).
  - Schemas forbid extra keys, so personal data cannot come back through the model output
    either.

## Consequences

- **Completeness depends on capture.** The worker must photograph every cohort page that got
  new entries that week, since a skipped page cannot be detected from images alone. The
  review page lists which pages contributed so the reviewer can confirm.
- **One reading mode: the whole page.** Tiles (aligned 4-row strips) were benchmarked first
  and removed on 2026-10-30: users photograph the whole register, and full-page accuracy is
  what needs measuring. Removing them also removed template alignment from the reader.
- **The benchmark measures integrity, not just OCR.** An oracle reviewer fixes only what was
  flagged. Any value still wrong after approval is an error that would have reached DHIS2
  unflagged; `value_accuracy_after_review` reports this.
- **Synthetic data comes first.** Scenarios are rendered on the real blank R-06 with
  handwriting fonts and photo effects, with injected edge cases. Real hand-filled, photographed
  pages will replace them as they become available.
- **RI v2 values not in the register stay absent from the payload:** inventory, missed
  opportunities and Unique Number of Forms.
- **First results:** synthetic scenario seed-1, 3 weeks, 5 page photos across scan, good and
  poor tiers.
  - **qwen3.5:4b in tiles mode, with key-labelled columns:**
    - 98.6% of written dates read correctly
    - 0.2% of blank cells wrongly filled
    - every injected problem flagged
    - 94.3% of approved daily values correct
    - about 98 s per page
  - **Full-page reading and granite-vision-4.1-4b were clearly worse.** Granite invents rows
    and fills blank cells.
  - **What helped most:** naming each tile column by its JSON key, instead of the small
    rotated print, removed most errors where a date was filed under the neighbouring column.
  - **Remaining errors** are still column shifts. A schedule check would expose them, since a
    dose N recorded without dose N-1, or below its minimum age, is a likely misread. That is
    the next integrity rule to add.
- **Open:**
  - the Polio (J) mapping
  - a stricter "zero-dose" definition, if DHIS2 needs one (currently the under-1 band)
  - storing register-derived values separately from tally-sheet values in DHIS2 (separate data
    elements), decided later before any import
