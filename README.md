# handwritten-forms-vlm-bench

**Which vision model reads your handwritten form well enough to trust, and how would you know?**

A benchmark for reading photos of handwritten forms with local vision-language models (VLMs) and turning them into [DHIS2](https://dhis2.org) aggregate data. It measures what matters for that data, not only how close the text is: the daily values that would actually be reported, and whether every mistake was flagged for a person to check.

**Case study:** a paper immunisation register. Health workers fill it in daily. Instead of re-counting it onto tally sheets every week, they photograph the pages; a model reads them, code counts every dose by date into daily immunisation values, and a person reviews and approves. Nothing is ever sent to DHIS2 by this project.

## How it works

```mermaid
flowchart LR
    A[Page photo] --> B[VLM + JSON schema<br/>+ fixed prompt]
    B --> C[Reading:<br/>rows and dates]
    C --> D[Code counts doses<br/>into daily DHIS2 values]
    D --> E[Flags for anything<br/>doubtful]
    E --> F{Person reviews<br/>and approves}
    F --> G[dataValueSet JSON<br/>never sent]
    C -.with a truth file.-> S[Scores in the terminal<br/>and in Langfuse]
    D -.-> S
```

- **The model only reads.** One call per page photo, the same prompt and JSON schema every time (`register/prompts.py`), temperature 0 and a fixed seed. The schema has no fields for names, phones or villages, so they cannot come back.
- **Code does the counting.** Dates are parsed, ages computed from the birth date, and values mapped to DHIS2 data elements and category option combos from the data set metadata (`register/spec.py`, `register/aggregate.py`).
- **Anything doubtful is flagged and not counted** until a person resolves it.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), and optionally a self-hosted [Langfuse](https://langfuse.com/self-hosting).

```bash
cp .env.example .env        # Langfuse keys (optional), DHIS2 login (only for `metadata`)
uv sync
ollama pull qwen3.5:4b      # or any vision model Ollama can serve

uv run formbench --help     # every command
uv run formbench scores     # what every score means
```

Private inputs, all kept out of git:

- `data/private/samples/`: the blank register PDF
- `data/private/photos/`: your real photos
- `data/private/synthetic/`: generated scenarios
- `schemas/dataset/metadata.json`: the DHIS2 data set metadata (`formbench metadata --dataset <uid>`)

## Compare models on your own photos

**1. Put the photos and their truth side by side** in one folder (the folder is your dataset):

```text
data/private/photos/my-pages/
  page-1.jpg   page-1.json
  page-2.jpg   page-2.json
```

A truth file lists what is really written, one object per child row. Name each column by
its letter, its key, or both:

```json
{"cohort_month": "June", "cohort_year": "2026",
 "rows": [{"A": "1", "D": "7654", "G": "M", "birth_date": "03-06-26", "age": "1/12",
           "I": "05-06-26", "K": "05-06-26", "L": "17-07-26"}]}
```

- Column C holds two values: call them `birth_date` and `age`.
- Leave blank cells out. Name, guardian, phone and village columns are ignored.
- A plain list of rows works too. The month and year are only scored if you include them.
- A misspelled column stops the run and lists the valid ones.

**2. Run each model on the folder**, one run per model (or several `--engine` at once):

```bash
uv run formbench bench --folder data/private/photos/my-pages --engine ollama:qwen3.5:4b
uv run formbench bench --folder data/private/photos/my-pages --engine ollama:hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q8_0
```

**3. See every model side by side:**

```bash
uv run formbench compare --folder my-pages
```

Example output (one column per model):

```text
                                   qwen3.5:4b   granite-vision
Reading accuracy
  Overall accuracy ↑                    51.0%            ...
  Completeness ↑                        53.4%
  Reliability ↑                         63.3%
  Rows detected ↑                       98.0%
  Processing success rate ↑            100.0%
DHIS2 reporting
  Reporting accuracy ↑                  29.0%
Performance
  Processing time (seconds per page) ↓  101.1
```

**4. See what one model got wrong on one photo,** cell by cell (the truth is found next to it):

```bash
uv run formbench eval --engine ollama:qwen3.5:4b --image data/private/photos/my-pages/page-1.jpg
```

No truth yet? Run `eval` on the photo anyway, then correct the reading it saves into a truth
file.

**Options:** `--max-side 4096` sends a larger image, `--think` lets thinking models think, and
`VLM_TIMEOUT=900` in `.env` allows more seconds per call.

## What the scores mean

A **cell** is one value in one row and column. It is right when the model put the same value
under the same column. Dates count as the same if they are the same day (`5/6/26` = `05-06-26`).

| Score | What it tells you |
| --- | --- |
| **Overall accuracy** | **The main score.** Of every cell that has a value on the page or in the output, the share that is right. |
| Completeness | Of the entries written on the page, the share captured correctly. Low: it misses or misreads. |
| Reliability | Of the values the model reported, the share that are correct. Low: it invents values or records them under the wrong column. |
| Rows detected | Child rows found, by serial number. |
| Processing success rate | Photos that produced a usable answer at all. |
| Reporting accuracy | Daily DHIS2 values counted from its reading that equal those counted from the truth. Stricter: needs birth date and sex right too. |
| Processing time | Seconds per page on this machine. |

Synthetic scenarios add three: reporting accuracy after review, error detection rate and review
workload (see below). `uv run formbench scores` prints all of them.

**Choosing a model:**

1. **Use the same photos and options for every model.**
2. **Rank by overall accuracy.** Then completeness and reliability tell you why one is worse:
   it misses things, or it makes things up.
3. **Only trust clear gaps.** Runs are repeatable (temperature 0, fixed seed), but with a few
   photos one cell is several points, so treat small gaps as a tie. More photos, smaller noise.
4. **Check the speed:** seconds per page times the pages you photograph each week.

In Langfuse, every folder is a dataset and every run one of its runs: open **Datasets →
\<folder\> → Runs** to compare models, with every photo's trace and scores.

## Synthetic test data

Synthetic registers are drawn on the real blank page, with known answers. Writers range from
neat to terrible, and photos from flat scans to dim and blurred:

```bash
uv run formbench template        # once: find the grid of the blank PDF
uv run formbench synth --name bad-poor --seed 25 --weeks 2 --messiness bad --quality poor
uv run formbench bench --folder data/private/synthetic/bad-poor --engine ollama:qwen3.5:4b
```

The same seed always gives the same photos. A scenario also runs the weekly workflow with a
stand-in reviewer who fixes only what was flagged, which adds **reporting accuracy after
review** (below 100% means wrong numbers nobody was warned about), **error detection rate**
and **review workload**.

## Weekly use

```bash
# In .env: REGISTER_GO_LIVE=YYYY-MM-DD (doses dated before it are never counted)
uv run formbench run --week 2026W44 --cover cover.jpg --pages p1.jpg p2.jpg --engine ollama:qwen3.5:4b
# -> results/register/<week>_<time>/review.html and decisions.yaml: resolve every flag, then
uv run formbench approve --run results/register/<week>_<time> --reviewer "Your Name"
# -> datavalueset.json. It is written for a person to import; nothing is sent to DHIS2.
```

## Using it for a different form

The project is built so that only one folder knows about the register:

| Keep as is | Replace for your form |
| --- | --- |
| `vlm.py`: one schema-constrained call to a local model, traced | `register/prompts.py`: your fields, schema and prompt |
| `metrics.py`: text similarity and row matching | `register/spec.py`: which field maps to which DHIS2 data element and category option combo |
| `evaluation/tracking.py`: Langfuse datasets, experiments, score definitions | `register/aggregate.py`: how readings become counts (a tally form may need no counting at all) |
| `evaluation/bench.py`, `photo.py`: running models over photos with truth files | `evaluation/scoring.py`: keep the cell scores; the DHIS2 value scores call your aggregation |
| `dhis2.py`: read-only metadata download | `evaluation/truth.py`: your column letters and names |

Start with 20–30 real photos and hand-made truth files in one folder, run `formbench bench` on it for each candidate model, and compare. Synthetic data helps you see failure modes; only real photos tell you how a model does on your form.

## Project layout

```text
src/formbench/
  cli.py          the formbench command; every option is defined here
  vlm.py          model call    metrics.py  generic scores    dhis2.py  read-only DHIS2
  register/       the register form: prompt and schema, reader, dates, counting
  evaluation/     scores, benchmark runs, the stand-in reviewer, Langfuse
  weekly/         draft, review page, approval, dataValueSet
  synthetic/      blank-page template, simulated registers, handwriting and photo effects
docs/adr/         why things are the way they are
```

## Results so far

qwen3.5:4b, whole-page photos, 6 synthetic scenarios (27 photos, neat scans to terrible handwriting):

- **Overall accuracy:** 51%
- **Reporting accuracy after review:** 30%, with an error detection rate of 52%
- **Main failure:** dates read correctly but filed under the neighbouring narrow column

Details in [ADR 0005](docs/adr/0005-register-evaluation-metrics.md) and [ADR 0007](docs/adr/0007-fewer-plainly-named-scores.md).

## Rules this project keeps

- **DHIS2 is read-only.** Metadata and org unit lookups only; the payload is written to a file.
- **A person approves every week,** even with no flags.
- **Private data stays local.** Photos, the blank register, the metadata snapshot, synthetic data and results are never committed.
- **Tests need nothing running.** No Ollama, Langfuse or network: `uv run pytest -q`.

Design decisions: [ADR 0003](docs/adr/0003-vlm-structured-output-and-thinking.md) structured output, [0004](docs/adr/0004-register-to-daily-aggregates.md) register to daily values, [0005](docs/adr/0005-register-evaluation-metrics.md) scores, [0006](docs/adr/0006-benchmark-runs-as-langfuse-experiments.md) Langfuse experiments, [0007](docs/adr/0007-fewer-plainly-named-scores.md) the scores and truth files.

## License

MIT
