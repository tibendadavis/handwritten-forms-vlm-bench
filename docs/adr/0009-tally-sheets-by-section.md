# ADR 0009: Tally sheets, read one section photo at a time

- **Status:** Accepted
- **Date:** 2026-11-03

## Context

Besides registers (one row per person), programmes report on tally sheets: totals written one
digit per box, in sections framed by corner squares so each can be photographed on its own.
We want to know how well local models read them, with the same scores, before any reporting.

## Decision

**A form is described once, in data.** `form.json` lists sections and their fields, each a
count, digits, text, date, choice or mark. It is kept out of git with the photos, since it
holds the form's own wording. Nothing about a particular form is in code.

**One photo per section.** Photos are named `<sheet>_<section>.jpg`; the model is asked only
for that section's fields, with a schema that allows nothing else. The section comes from the
file name, standing in for an app that tells the user which section to photograph.

**One truth file per sheet.** It holds every field of the filled sheet, as people write it;
each section photo is scored on its own fields. Counts compare as numbers and an empty box
equals zero ("leave empty if none"), dates as dates, choices ignoring punctuation.

**Same scores, same tooling.** Overall accuracy, completeness, reliability, processing success
rate and time come from the same counts as for registers (rows detected does not apply); each
folder is a Langfuse dataset, runs are saved under `results/tally/bench` and shown by
`formbench compare`.

**One prompt per form**, versioned in Langfuse as `tally-<form id>`, with `{{section}}`,
`{{title}}` and `{{fields}}` filled in per photo, so improving the instructions improves every
section at once.

## Consequences

- A new tally form needs a `form.json`, photos and truth files; no code.
- Reading the section code printed on the form, instead of trusting the file name, is not
  tested yet.
- The model call with its timeout handling and the Langfuse prompt loader are shared by both
  kinds of form (`vlm.read_json`, `prompt_store`).
