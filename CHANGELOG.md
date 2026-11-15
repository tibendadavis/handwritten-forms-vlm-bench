# Changelog

## [0.2.0] - 2026-11-15

### Changed
- formbench now only measures how well a local vision model reads handwritten forms, with six
  commands: `check`, `run`, `compare`, `breakdown`, `rescore`, `scores`.
- Three kinds of form, each described by a `form.json`: patient register,
  computer-vision-friendly form (section or whole-page photos) and monthly summary report.
- Photos are sent exactly as taken; prompts are versioned in Langfuse; every run is a Langfuse
  experiment and is saved locally.

### Removed
- Reporting to external systems, the weekly review workflow and synthetic data.

## 0.1.0
- First version: register and tally-sheet benchmarks, with reporting and a weekly review
  workflow.
