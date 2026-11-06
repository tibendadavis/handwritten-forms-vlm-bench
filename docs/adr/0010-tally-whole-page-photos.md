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

**Photos are sent as they are, never enlarged by code.** Enlarging small page photos helped
on the samples: at their size a digit box is about 20 px, and the 4B model lost track of which
boxes belong to which field (joining two fields' digits, dropping whole rows). Enlarged 2× it
went from 69% to 85% of cells right and the 9B model from 87% to 90%. But a benchmark should
measure the photo as it was taken, so code does not resize or re-compress it; whoever wants to
test an enlarged or cropped photo makes it and puts it in its own folder.

**Its own prompt**, `tally-<form id>-page` in Langfuse, with `{{fields}}`. It tells the model
to find each section by its heading and each field by its row and column, and to read only a
field's own group of boxes. It was tuned on enlarged page photos; wording that helped the 4B
model made no difference to the 9B model.

## Consequences

- With the 9B model on a laptop a sample page takes about 20 s as taken (40 s enlarged 2×),
  against about 10 s for each of its four section photos.
- Photos straight from a phone camera (about 3000 × 4000 px) should not need enlarging; the
  samples were small because they had been through a chat app.
- Which page a photo shows comes from its file name, as the section does for section photos.
