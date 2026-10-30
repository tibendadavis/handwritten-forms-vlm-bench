"""formbench: pick the vision model that reads a handwritten DHIS2 form best.

vlm.py         one schema-constrained call to a local model (Ollama), traced in Langfuse
metrics.py     generic text, list and tree similarity (CER, ANLS, Hungarian matching, TED)
dhis2.py       read-only DHIS2 client: data set metadata, org unit lookup
register/      the register: what to ask the model, and how readings become counts
evaluation/    readings vs truth: scores, benchmark runs, Langfuse datasets and experiments
weekly/        the weekly workflow: draft, human review, approval, dataValueSet (never sent)
synthetic/     synthetic handwritten registers with known answers
cli.py         the `formbench` command
files.py       file names and JSON files
"""
