"""The formbench commands: check, run, compare, breakdown, rescore and scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markup import escape

from formbench.forms import load_form
from formbench.forms.base import FormError, ReadOptions
from formbench.model import DEFAULT_MAX_SIDE
from formbench.report import (
    breakdown_table,
    check_table,
    compare_tables,
    score_table,
    scores_table,
)
from formbench.runner import prepare, rescore, run_label, run_model

RESULTS = Path("results")
console = Console()


def say(message: str, style: str | None = None) -> None:
    """Print a message as plain text: brackets in it are not markup."""
    console.print(escape(message), style=style, highlight=False)


def warn(message: str) -> None:
    say(message, "yellow")


def fail(message: str) -> int:
    say(message, "red")
    return 1


def check(args: argparse.Namespace) -> int:
    """Check a folder without a model or Langfuse: the form, every truth file, and the photos
    a run would read. Exit code 1 when anything is wrong."""
    form = load_form(args.folder, args.form)
    say(f"Form: {form.name} ({form.kind}, id {form.id})")
    errors = form.check_truths(args.folder)
    for error in errors:
        say(error, "red")
    if errors:
        return fail(f"{len(errors)} truth file(s) cannot be used.")
    items, skipped = form.items(args.folder, args.pages)
    console.print(check_table(items))
    for photo in skipped:
        warn(f"skipped: {photo}")
    if not items:
        return fail(f"No photo in {args.folder} has a truth file.")
    return 0


def run(args: argparse.Namespace) -> int:
    """Benchmark each model on a folder, one Langfuse experiment per model."""
    from langfuse import get_client

    form = load_form(args.folder, args.form)
    options = ReadOptions(think=args.think, max_side=args.max_side)
    langfuse = get_client()
    plan = prepare(
        langfuse, args.folder, form, pages=args.pages, options=options, prompt_choice=args.prompt
    )
    say(form.name)
    say(f"Prompt: {plan.prompt_name} v{plan.version}")
    columns = []
    for model in args.model:
        say(f"Running {model} on {args.folder.name}...")
        record = run_model(plan, model, langfuse, args.out)
        if record["failed"]:
            warn(f"{len(record['failed'])} photo(s) failed: {', '.join(record['failed'])}")
        if record["skipped"]:
            warn(f"{len(record['skipped'])} photo(s) skipped: {', '.join(record['skipped'])}")
        for empty in record["empty_readings"]:
            parts = ", ".join(str(p) for p in empty["parts"])
            warn(f"empty reading: {empty['file']} ({parts}) came back blank; the truth has values")
        if record["dataset_run_url"]:
            say(f"Langfuse: {record['dataset_run_url']}")
        say(f"Saved to {record['path']}")
        columns.append((run_label(record["settings"]), record["metrics"]))
    console.print(score_table(f"{args.folder.name}: this run", columns))
    say(f"All runs on this folder side by side: formbench compare --folder {args.folder.name}")
    return 0


def load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    """Saved runs; files that are not run results are passed over with a note."""
    runs = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and {"settings", "counts"} <= set(data):
            runs.append(data)
        else:
            warn(f"not a formbench run, left out: {path}")
    return runs


def compare(args: argparse.Namespace) -> int:
    """One table per folder with every run on it; runs with the same label are pooled."""
    runs = load_runs(args.runs or sorted(RESULTS.glob("*.json")))
    if args.folder:
        runs = [r for r in runs if r["settings"]["folder"] == args.folder]
    if not runs:
        return fail("No saved runs to compare (formbench run saves them).")
    for table in compare_tables(runs):
        console.print(table)
    return 0


def breakdown(args: argparse.Namespace) -> int:
    runs = load_runs([args.run])
    if not runs:
        return 1
    console.print(breakdown_table(runs[0], args.by))
    return 0


def rescore_runs(args: argparse.Namespace) -> int:
    """Re-score saved runs from their photos and the current truth files."""
    form = load_form(args.folder, args.form)
    columns = []
    for path in args.runs:
        run_record = rescore(path, args.folder, form)
        columns.append((run_label(run_record["settings"]), run_record["metrics"]))
        say(f"Re-scored {path}")
    console.print(score_table(f"Re-scored: {args.folder.name}", columns))
    return 0


def scores(args: argparse.Namespace) -> int:
    console.print(scores_table())
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="formbench", description="Measure how well a local VLM reads photos of forms."
    )
    commands = top.add_subparsers(dest="command", required=True)

    def folder_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--folder", type=Path, required=True, help="the photo folder")
        p.add_argument("--form", type=Path, help="form.json (default: in the folder or above)")

    p = commands.add_parser("check", help="check a folder's form, truth files and photo names")
    folder_options(p)
    p.add_argument("--pages", action="store_true", help="whole-page photos (cv-form)")
    p.set_defaults(handler=check)

    p = commands.add_parser("run", help="benchmark models on a folder (needs Ollama, Langfuse)")
    folder_options(p)
    p.add_argument("--model", action="append", required=True, help="an Ollama model tag")
    p.add_argument("--pages", action="store_true", help="whole-page photos (cv-form)")
    p.add_argument("--prompt", help="Langfuse prompt version number or label (production)")
    p.add_argument("--think", action="store_true", help="let the model think first")
    p.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE, help="longest photo side")
    p.add_argument("--out", type=Path, default=RESULTS, help="where results files go")
    p.set_defaults(handler=run)

    p = commands.add_parser("compare", help="compare saved runs, one table per folder")
    p.add_argument("runs", nargs="*", type=Path, help="run files (default: results/*.json)")
    p.add_argument("--folder", help="only runs on this folder (its name)")
    p.set_defaults(handler=compare)

    p = commands.add_parser("breakdown", help="accuracy of one run by group or field type")
    p.add_argument("run", type=Path)
    p.add_argument("--by", choices=("group", "type"), default="group")
    p.set_defaults(handler=breakdown)

    p = commands.add_parser("rescore", help="re-score saved runs from the current truth files")
    p.add_argument("runs", nargs="+", type=Path)
    folder_options(p)
    p.set_defaults(handler=rescore_runs)

    p = commands.add_parser("scores", help="what every score means")
    p.set_defaults(handler=scores)
    return top


def main(argv: list[str] | None = None) -> int:
    """Entry point of the formbench console script."""
    load_dotenv(find_dotenv(usecwd=True))
    args = parser().parse_args(argv)
    try:
        return args.handler(args)
    except FormError as error:
        return fail(str(error))
    except SystemExit as stop:
        if isinstance(stop.code, str):  # a run or prompt that cannot go on, with why
            return fail(stop.code)
        raise


if __name__ == "__main__":
    sys.exit(main())
