"""Test one register photo with one model and see exactly where it goes wrong.

    register-eval --scenario data/private/synthetic/<name> --photo week-02/page-....jpg --engine ollama:<tag>
    register-eval --image photo.jpg --truth truth.json --engine ollama:<tag>
    register-eval --image photo.jpg --engine ollama:<tag>          # no truth: transcription only

A truth file is a page reading: {"cohort_month", "cohort_year", "rows": [{serial, card_number,
sex, birth_date, age, bcg, polio, opv0, ...}]}. For a real photo, run once without --truth,
correct the saved reading by hand, and use it as the truth from then on.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.table import Table

from formbench.register_schema import ROW_FIELDS
from formbench.register_scoring import same_value
from formbench.schema_gen import slugify

OUT_DIR = Path("results/register/eval")


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
            raise SystemExit("register-eval: --scenario needs --photo (path inside the scenario)")
        truth = json.loads((args.scenario / "truth.json").read_text())
        for week in truth["weeks"]:
            for photo in week["photos"]:
                if photo["file"] == args.photo:
                    return args.scenario / photo["file"], photo["reading"]
        files = [p["file"] for w in truth["weeks"] for p in w["photos"]]
        raise SystemExit(f"register-eval: {args.photo} not in scenario; photos: {', '.join(files)}")
    if not args.image:
        raise SystemExit(
            "register-eval: give --image (and optionally --truth) or --scenario and --photo"
        )
    truth = json.loads(args.truth.read_text()) if args.truth else None
    return args.image, truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Test one register photo with one model.")
    parser.add_argument("--engine", required=True, help="ollama:<model-tag>")
    parser.add_argument("--image", type=Path, help="a register page photo")
    parser.add_argument("--truth", type=Path, help="true page reading (JSON) for --image")
    parser.add_argument("--scenario", type=Path, help="synthetic scenario folder")
    parser.add_argument("--photo", help="photo path inside the scenario, e.g. week-02/page-x.jpg")
    parser.add_argument("--mode", choices=["full", "tiles"], default="full")
    parser.add_argument("--max-side", type=int, help="longest image side sent in full mode (default 3072)")  # fmt: skip
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    kind, _, model = args.engine.partition(":")
    if kind != "ollama" or not model:
        raise SystemExit("register-eval: --engine must be 'ollama:<model-tag>'")
    image, truth = _truth_for(args)
    load_dotenv(find_dotenv(usecwd=True))

    from langfuse import get_client

    from formbench.register_bench import read_scored_page
    from formbench.register_template import load_template

    template, layout = load_template()
    langfuse = get_client()
    settings = {"engine": args.engine, "image": str(image), "mode": args.mode, "max_side": args.max_side,
                "think": args.think, "has_truth": truth is not None}  # fmt: skip
    try:
        with langfuse.start_as_current_span(name="register-eval", metadata=settings):
            langfuse.update_current_trace(name=f"register-eval {image.name}", metadata=settings, tags=["register-eval", args.mode])  # fmt: skip
            result = read_scored_page(image, truth, model, mode=args.mode, template=template,
                                      layout=layout, think=args.think, max_side=args.max_side)  # fmt: skip
            for name, value in (result.get("scores") or {}).items():
                if value is not None:
                    langfuse.score_current_trace(name=name, value=float(value))
            trace_id = langfuse.get_current_trace_id()
    finally:
        langfuse.flush()

    console = Console()
    reading = result["reading"]
    console.print(f"[bold]{image.name}[/bold] with {model} ({result['mode']}): "
                  f"{len(reading['rows'])} rows read in {result['latency_s']} s; JSON valid: {result['json_valid']}")  # fmt: skip
    for note in result["notes"]:
        console.print(f"[yellow]{note}[/yellow]")
    if truth is not None:
        diffs = differences(reading, truth)
        table = Table(title=f"{len(diffs)} cell(s) read differently from the truth")
        for column in ("Row", "Field", "Truth", "Read"):
            table.add_column(column)
        for serial, field, want, got in diffs:
            table.add_row(str(serial), field, "—" if want is None else str(want), "—" if got is None else str(got))  # fmt: skip
        console.print(table)
        scores = Table(title="Scores (also on the Langfuse trace)", show_header=False)
        for name, value in result["scores"].items():
            shown = "n/a" if value is None else (f"{value:.4f}" if name in ("cer", "wer") else f"{value:.1%}")  # fmt: skip
            scores.add_row(name, shown)
        console.print(scores)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"{stamp}_{slugify(image.stem)}_{slugify(args.engine)}.json"
    record = {"settings": settings, "trace_id": trace_id, "latency_s": result["latency_s"],
              "json_valid": result["json_valid"], "notes": result["notes"], "scores": result.get("scores"),
              "reading": reading, "raw": result["raw"]}  # fmt: skip
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    console.print(f"Saved to {path}" + ("" if truth else " (its 'reading' can be corrected into a truth file)"))  # fmt: skip


if __name__ == "__main__":
    main()
