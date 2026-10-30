"""Test one register photo with one model and see exactly where it goes wrong.

    formbench eval --scenario data/private/synthetic/<name> --photo week-02/page-....jpg --engine ollama:<tag>
    formbench eval --image photo.jpg --truth truth.json --engine ollama:<tag>
    formbench eval --image photo.jpg --engine ollama:<tag>          # no truth: reading only

A truth file is a page reading: {"cohort_month", "cohort_year", "rows": [{serial, card_number,
sex, birth_date, age, bcg, polio, opv0, ...}]}. For a real photo, run once without --truth,
correct the saved reading by hand, and use it as the truth from then on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from formbench.evaluation.scoring import same_value
from formbench.files import slugify, stamp, write_json
from formbench.register.prompts import ROW_FIELDS


def differences(predicted: dict[str, Any], truth: dict[str, Any]) -> list[tuple[str, str, Any, Any]]:  # fmt: skip
    """(serial, field, truth, read) for every cell read differently; rows matched by serial."""
    out = []
    for key in ("cohort_month", "cohort_year"):
        if str(predicted.get(key) or "").strip().casefold() != str(truth.get(key) or "").strip().casefold():  # fmt: skip
            out.append(("header", key, truth.get(key), predicted.get(key)))
    read = {str(r.get("serial")): r for r in predicted["rows"]}
    for row in truth["rows"]:
        got = read.pop(str(row["serial"]), None)
        if got is None:
            out.append((row["serial"], "(whole row)", "present", "missing"))
            continue
        out += [(row["serial"], k, row.get(k), got.get(k)) for k in ROW_FIELDS if not same_value(row.get(k), got.get(k), k)]  # fmt: skip
    out += [(serial, "(whole row)", "absent", "invented") for serial in read]
    return out


def _truth_for(args: argparse.Namespace) -> tuple[Path, dict[str, Any] | None]:
    if args.scenario:
        if not args.photo:
            raise SystemExit("formbench eval: --scenario needs --photo (path inside the scenario)")
        truth = json.loads((args.scenario / "truth.json").read_text())
        for week in truth["weeks"]:
            for photo in week["photos"]:
                if photo["file"] == args.photo:
                    return args.scenario / photo["file"], photo["reading"]
        files = [p["file"] for w in truth["weeks"] for p in w["photos"]]
        raise SystemExit(
            f"formbench eval: {args.photo} not in scenario; photos: {', '.join(files)}"
        )
    if not args.image:
        raise SystemExit(
            "formbench eval: give --image (and optionally --truth) or --scenario and --photo"
        )
    truth = json.loads(args.truth.read_text()) if args.truth else None
    return args.image, truth


def command(args: argparse.Namespace) -> None:
    from langfuse import get_client

    from formbench.cli import model_tag
    from formbench.evaluation.bench import read_scored_page, score_table
    from formbench.register.spec import load_spec

    model = model_tag(args.engine)
    image, truth = _truth_for(args)
    spec = load_spec(json.loads(args.metadata.read_text()))

    langfuse = get_client()
    settings = {"engine": args.engine, "image": str(image), "max_side": args.max_side,
                "think": args.think, "has_truth": truth is not None}  # fmt: skip
    try:
        with langfuse.start_as_current_span(name="register-eval", metadata=settings):
            langfuse.update_current_trace(name=f"register-eval {image.name}", metadata=settings, tags=["register-eval"])  # fmt: skip
            result = read_scored_page(image, truth, model, spec, think=args.think, max_side=args.max_side)  # fmt: skip
            for name, value in (result.get("scores") or {}).items():
                langfuse.score_current_trace(name=name, value=float(value))
            trace_id = langfuse.get_current_trace_id()
    finally:
        langfuse.flush()

    console = Console()
    reading = result["reading"]
    console.print(f"[bold]{image.name}[/bold] with {model}: "
                  f"{len(reading['rows'])} rows read in {result['latency_s']} s; JSON valid: {result['json_valid']}")  # fmt: skip
    for error in result["errors"]:
        console.print(f"[yellow]{error}[/yellow]")
    if truth is not None:
        diffs = differences(reading, truth)
        table = Table(title=f"{len(diffs)} cell(s) read differently from the truth")
        for column in ("Row", "Field", "Truth", "Read"):
            table.add_column(column)
        for serial, field, want, got in diffs:
            table.add_row(str(serial), field, "—" if want is None else str(want), "—" if got is None else str(got))  # fmt: skip
        console.print(table)
        console.print(score_table("Scores (also on the Langfuse trace)", [(args.engine, result["scores"])]))  # fmt: skip

    path = args.out / f"{stamp()}_{slugify(image.stem)}_{slugify(args.engine)}.json"
    record = {"settings": settings, "trace_id": trace_id, "latency_s": result["latency_s"],
              "json_valid": result["json_valid"], "errors": result["errors"], "scores": result.get("scores"),
              "reading": reading, "raw": result["raw"]}  # fmt: skip
    write_json(path, record)
    console.print(f"Saved to {path}" + ("" if truth else " (its 'reading' can be corrected into a truth file)"))  # fmt: skip
