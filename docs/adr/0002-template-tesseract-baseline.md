# ADR 0002: Template OCR with Tesseract as the tier-0 baseline

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

The benchmark asks whether a small VLM is worth using on handwritten DHIS2 forms. Without a
non-VLM baseline there is nothing to beat. The baseline should be cheap, deterministic, run
offline and need no training data.

We generate the forms ourselves (`make-form`), so we know exactly where every field is printed.

## Decision

Use **template OCR**: align the photo to the known page, crop each cell by its layout box and
read it with Tesseract.

1. **Alignment.** Four solid black corner markers are printed on every form. `align.py` finds
   them (Otsu threshold, with an adaptive-threshold fallback for uneven light). It then applies
   a perspective warp from the photo onto the template's page size (A4 at 300 DPI).
2. **Empty-cell detection before OCR.** Each cell is cropped about 12% inside its border and
   binarised against the page's paper level. A cell with less than 1% ink is `null` and never
   reaches Tesseract. This stops Tesseract from turning smudges or border remnants into digits
   on blank cells, which is the error the benchmark cares about most (`hallucination_rate`).
3. **Constrained recognition.** Count cells use `--psm 7` (single line) with a `0-9`
   whitelist, and fall back to `--psm 8` (single word) when psm 7 returns nothing. In testing,
   psm 7 often returned an empty string for a lone digit. Header boxes (facility, period) are
   read as free text.
4. **Same output shape as the VLM**, plus per-field Tesseract confidence (0–1). Both engines go
   through the same scoring and Langfuse reporting.

## Consequences

- Fast (about 2 s per form on a laptop) and fully deterministic, with no model download.
- It only works on **our printed template**. It cannot read an arbitrary form, a different
  layout or a form without markers. The VLM path has no such limit, and that difference is
  part of what the benchmark measures.
- Tesseract is trained on print, not handwriting. Expect clearly worse results on real
  handwriting than on the printed digits used in tests. Even on clean printed digits a
  20-form sweep got 516/520 right; all misses were `77` read as `7`.
- Alignment assumes the page is upright and all four markers are visible. A photo rotated by
  90°/180° or with a corner cut off fails with `MarkersNotFound` rather than a wrong reading.
- The empty-cell threshold (1% ink) and margin (12%) are fixed constants in `ocr_baseline.py`.
  Tune them only with a dataset version bump, so results stay comparable.
