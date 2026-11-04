# ADR 0010: Tally sheets, whole-page photos as an alternative

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

ADR 0009 reads a tally sheet one section photo at a time. Taking a photo per section is slow
for the person holding the phone, so we want to know what is lost when they photograph each
whole page instead, on the same sheets and the same truth files.

Whole-page photos sent through phone chat apps arrive at about 900 × 1250 px. A digit box is
then about 20 px wide.

## Decision

**Same folder, same truth.** `<sheet>.jpg` is page 1 of a sheet and `<sheet>_p2.jpg` page 2.
`form.json` lists the sections printed on each page (`pages`). `formbench tally --pages` runs
the page photos instead of the section photos; each page is scored on its sections exactly as
section photos are, so `compare` puts the two side by side.

**One call per page.** The model gets every field of the page's sections in one schema,
grouped by section in the prompt. On six sample sheets this was more accurate than asking for
one section at a time on the same page photo, and half the time (one image instead of four).

**Small page photos are enlarged** to 2560 px on the long side before they are sent. At the
size they arrive, the 4B model lost track of which boxes belong to which field (joining two
fields' digits, dropping whole rows): 58% of cells right; enlarged, 82%. The 9B model went
from 87% to 90%. Enlarging adds no detail, but gives the model more image tokens per digit.
Section photos are not enlarged, so earlier runs stay comparable.

**Its own prompt**, `tally-<form id>-page` in Langfuse, with `{{fields}}`. It tells the model
to find each section by its heading and each field by its row and column, and to read only a
field's own group of boxes. Wording that helped the 4B model on whole pages made no difference
to the 9B model.

## Consequences

- With the 9B model on a laptop a whole page takes about 40 s, about the same as its four
  section photos (about 10 s each): the time saved is the person's, not the model's.
- Enlarging is fixed at 2560 px; a photo larger than that is sent as it is (up to
  `--max-side`).
- Which page a photo shows comes from its file name, as the section does for section photos.
