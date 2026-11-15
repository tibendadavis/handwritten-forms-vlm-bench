# Design notes

Key decisions, and why.

## Measuring

- **Photos are sent as taken.** A JPEG or PNG that is upright and within `--max-side` is sent
  byte for byte; anything else is turned upright, shrunk and sent as JPEG quality 90. Never
  enlarged, never re-compressed when it can go as is. Re-encoding changed results: in
  earlier runs one section photo was read 29/30 correctly as JPEG and 0/30 as the original
  PNG. The benchmark measures the photos people take; edited photos go in their own folder.
- **Temperature 0 and a fixed seed (42).** The same photo and prompt give the same reply, so a
  change in scores comes from a change in model, prompt or photo, not chance.
- **No retries.** A retry of a deterministic call repeats the same answer and only multiplies
  the wait. A timeout is scored as a photo not read; any other error stops loudly.
- **`max_concurrency=1`.** One call at a time, so processing times are fair on one machine.
- **One call per whole page.** A page photo is read with one schema covering all its sections:
  that was more accurate than one call per section on the same photo, and faster.

## Scoring

- **Counts are summed before rates are taken**, over photos and over pooled runs. An average
  of per-photo rates would let a photo with two cells weigh as much as one with two hundred.
- **Register rows are paired by content** (mean ANLS over the filled columns, then a minimum-
  cost assignment), so one invented or missing row does not shift every row below it.
- **Failed photos are scored as not read.** A photo whose read raised (and so has no output in
  the experiment) is kept in the results file with an empty reading: every truth cell counts
  as missed and it lowers the processing success rate. A run where every photo fails stops.
- **Empty-reading warning.** A model can now and then return a whole section blank on a photo
  it reads correctly once re-saved. A blank section is easy to detect and worth flagging, so
  runs list the parts that came back blank while the truth has values there. Only usable
  (valid JSON) replies are checked; a timeout or bad reply is already reported as such.

## Privacy

- **Personal fields are kept out of the schema.** The register schema has no name, guardian,
  phone or village field and forbids extra keys, so the model cannot return them; personal and
  unread columns in truth files are dropped when the truth is loaded. None of it reaches
  Langfuse or results files.
- `data/private/`, `results/` and `.env` are git-ignored; fixtures use made-up content.

## Prompts

- **Versioned in Langfuse.** The text in code is only version 1, saved with the `production`
  label the first time a prompt name is used. After that, prompts are edited in Langfuse and
  each run records the version it used (settings, run name, trace metadata, and every
  generation is linked to it).
- **A missing version never falls back to another.** Asking for a version or label that does
  not exist stops the run: results must say exactly what was measured.
- The code defaults are long single-line paragraphs in Langfuse; in code they are split into
  string pieces so every source line fits in 100 characters, and stay identical word for word.

## Smaller choices

- **Truth files are checked one by one.** Besides the protocol's methods, each kind has
  `check_truths(folder)`, so `formbench check` reports the first error of every bad file, not
  only the first bad file. Skipped photos are warnings: `check` still exits 0 with them.
- **Photos of the other cv-form mode** are passed over silently when their sheet has a truth
  file. Without one, the photo fits neither mode for a known sheet and is listed as skipped,
  with the naming rule.
- **The register needs a `serial` column** (rows are found by serial number), and its header
  holds `cohort_month` and `cohort_year`; truth files may also call them `month` and `year`.
- **cv-form section letters are single letters**, because photos are named by them.
- **File names:** the model slug also drops leading and trailing `_`.
