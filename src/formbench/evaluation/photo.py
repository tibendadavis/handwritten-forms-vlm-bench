"""Test one register photo with one model and see exactly which cells it gets wrong.

    formbench eval --image photo.jpg --engine ollama:<tag>          # truth found next to it
    formbench eval --image photo.jpg --truth truth.json --engine ollama:<tag>
    formbench eval --scenario data/private/synthetic/<name> --photo week-02/page-....jpg --engine ollama:<tag>

The truth is found next to the photo (photo.json or photo.truth.json) unless --truth names it;
its format is in evaluation/truth.py. Without a truth, the reading is saved so it can be
corrected by hand into one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from formbench.evaluation.scoring import HEADER, same_value
from formbench.evaluation.truth import load_truth, truth_file
from formbench.files import slugify, stamp, write_json
from formbench.register.prompts import ROW_FIELDS


def differences(predicted: dict[str, Any], truth: dict[str, Any]) -> list[tuple[str, str, Any, Any]]:  # fmt: skip
    """(serial, field, truth, read) for every cell read differently; rows matched by serial."""
    out = []
    for key in (k for k in HEADER if k in truth):
        if str(predicted.get(key) or "").strip().casefold() != str(truth.get(key) or "").strip().casefold():  # fmt: skip
            out.append(("header", key, truth.get(key), predicted.get(key)))
    read = {str(r.get("serial")).strip(): r for r in predicted["rows"]}
    for row in truth["rows"]:
        got = read.pop(str(row["serial"]).strip(), None)
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
    path = args.truth or truth_file(args.image)
    return args.image, load_truth(path) if path else None


def command(args: argparse.Namespace) -> None:
    from langfuse import get_client

    from formbench.cli import model_tag
    from formbench.evaluation.bench import read_scored_page, score_table
    from formbench.evaluation.tracking import score_configs
    from formbench.register.prompts import load_prompt
    from formbench.register.reader import prompt_version
    from formbench.register.spec import load_spec

    model = model_tag(args.engine)
    image, truth = _truth_for(args)
    spec = load_spec(json.loads(args.metadata.read_text()))

    langfuse = get_client()
    try:
        configs = score_configs(langfuse) if truth is not None else {}
    except Exception:  # Langfuse not reachable: scores are still printed and saved
        configs = {}
    prompt = load_prompt(langfuse, "page", args.prompt)
    settings = {"engine": args.engine, "image": str(image), "max_side": args.max_side,
                "think": args.think, "has_truth": truth is not None,
                "prompt_version": prompt_version(prompt)}  # fmt: skip
    try:
        with langfuse.start_as_current_span(name="register-eval", metadata=settings):
            langfuse.update_current_trace(name=f"register-eval {image.name}", metadata=settings, tags=["register-eval"])  # fmt: skip
            result = read_scored_page(image, truth, model, spec, think=args.think,
                                      max_side=args.max_side, configs=configs, prompt=prompt)  # fmt: skip
            for name, value in (result.get("scores") or {}).items():
                langfuse.score_current_trace(name=name, value=float(value), config_id=configs.get(name))  # fmt: skip
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
