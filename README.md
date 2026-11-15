# formbench

Measures how well a local vision model reads photos of handwritten paper forms.

You give it photos plus a **truth file** (what is really written on each form). It sends each
photo to a model in [Ollama](https://ollama.com), compares the answer with the truth field by
field, and reports scores. Every run and prompt is tracked in [Langfuse](https://langfuse.com).

Three kinds of form are supported:

- **`register`**: a table with one row per patient, read one photo per page.
- **`cv-form`**: boxed sections, one character per box, choices marked by filling a circle.
  Photographed one section per photo, or one whole page per photo (`--pages`).
- **`summary`**: a monthly summary report, tables of totals written freehand in cells, read
  one photo per page.

## Setup

You need [uv](https://docs.astral.sh/uv/), Ollama with your models already pulled, and a
Langfuse server.

```sh
git clone https://github.com/tibendadavis/handwritten-forms-vlm-bench.git
cd handwritten-forms-vlm-bench
uv sync                 # installs Python 3.11 and the dependencies
cp .env.example .env    # add your Langfuse keys and host
ollama pull qwen3.5:9b  # or any vision model you want to measure
```

## Quick start

1. **Add your data** (it stays on your machine and is never committed). Start each
   `form.json` from the example in `tests/fixtures/<kind>/form.json`:
   ```
   data/private/
     register/
       form.json              your register-form description
       <folder>/page-01.jpg   + page-01.json (the truth)
     cv-form/
       form.json              your cv-form description
       <folder>/sheet-01_A.jpg, sheet-01_B.jpg ...   section photos
                sheet-01.jpg, sheet-01_p2.jpg         whole-page photos (page 1, page 2)
                sheet-01.json                         truth for the whole sheet
     summary/
       form.json              your summary-form description
       <folder>/page-01.jpg   + page-01.json (the truth)
   ```
2. **Check it** (no model or Langfuse needed). Fix anything it reports:
   ```sh
   uv run formbench check --folder data/private/cv-form/<folder>          # add --pages for page photos
   uv run formbench check --folder data/private/register/<folder>
   uv run formbench check --folder data/private/summary/<folder>
   ```
3. **Run** (use model tags exactly as `ollama list` shows them):
   ```sh
   uv run formbench run --folder data/private/cv-form/<folder> --model qwen3.5:9b --model qwen3.5:4b
   uv run formbench run --folder data/private/register/<folder> --model qwen3.5:9b
   uv run formbench run --folder data/private/summary/<folder> --model qwen3.5:9b
   ```
4. **Compare and dig in:**
   ```sh
   uv run formbench compare --folder <folder>
   uv run formbench breakdown results/<file>.json      # add --by type for field types
   ```

## Commands

| Command | What it does |
|---|---|
| `check --folder PATH [--pages]` | Validates the form, truth files and photo names. |
| `run --folder PATH --model TAG ...` | Runs a benchmark. Options: `--pages`, `--prompt N\|LABEL`, `--think`, `--max-side N`, `--out DIR`. |
| `compare [RUN.json ...] [--folder NAME]` | Shows saved runs side by side. Runs with the same settings are pooled. |
| `breakdown RUN.json [--by group\|type]` | Shows where one run went wrong. |
| `rescore RUN.json ... --folder PATH` | Re-scores saved runs after you fix a truth file. No model needed. |
| `scores` | Explains every score. |

`check`, `run` and `rescore` read `form.json` from the photo folder, or from the folder above
it. Use `--form PATH` to point to a different one.

## Scores

A **cell** is one field, or one row × column value in a register. Cells that are blank in both
the truth and the model's answer are not counted.

| Score | Means |
|---|---|
| **Overall accuracy** ↑ (main score) | Share of cells read correctly |
| Completeness ↑ | Share of what is written that was captured. Low: the model misses things. |
| Reliability ↑ | Share of what the model reported that is true. Low: the model invents values. |
| Rows detected ↑ | Registers only: share of rows found, by serial number |
| Processing success rate ↑ | Share of photos that gave a usable answer |
| Processing time ↓ | Seconds per photo |

Values are compared sensibly: dates as dates (`12/6/26` = `12-06-2026`), counts as numbers
(an empty count box is the same as 0), and text ignoring case and spacing.

## Truth files

- **Register:** `page-01.json` (or `page-01.truth.json`) next to `page-01.jpg`. It holds a list
  of rows, or `{"cohort_month": "7", "cohort_year": "2026", "rows": [...]}`. You can name a
  column by its key, label or letter (`bcg`, `BCG`, `I`, `BCG_I`). Column C holds two values,
  so name them `birth_date` and `age`. Personal columns (names, phone, village) are dropped
  automatically.
- **cv-form:** one `<sheet>.json` per filled sheet, holding field name → value for the whole
  sheet, with names as in `form.json`.
- **summary:** `page-01.json` next to `page-01.jpg`, holding field name → value for the whole
  page, with names as in `form.json` (`BCG (Bacillus Calmete-Guerin)_0-11 m_Facility`).
  `form.json` is a flat list of `fields` written like a cv-form's, with no sections; shaded
  cells are not listed. Fields left out of a truth file are blank, and an empty cell counts the
  same as 0. The reporting officer's name is dropped.
- Put numbers with leading zeros in quotes: `"0001"`.

## Prompts

Prompts live in Langfuse. The first run creates version 1 from the code. After that, edit
prompts in Langfuse (Prompts > name > new version). Run a specific version with `--prompt N`,
or move the `production` label to make that version the default. If the version you ask for
doesn't exist, the run stops; it never silently uses another version.

## Good to know

- **Photos are sent exactly as taken.** They are only rotated upright or shrunk when needed
  (longest side above 4032 px, or set `--max-side`). Re-saving a photo can change results,
  so put edited photos in their own folder.
- **Privacy:** `data/private/`, `results/` and `.env` are git-ignored and must never be
  committed. The model is never asked for names, phone numbers or villages.
- Design decisions and why: [docs/design.md](docs/design.md).

## Development

```sh
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

Tests run offline, with no Ollama or Langfuse needed.
