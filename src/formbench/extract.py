from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from jsonschema import Draft202012Validator
from rich.console import Console
from rich.table import Table

from formbench.schema_gen import slugify
from formbench.scoring import score
from formbench.vlm import PROMPT_VERSION, SEED


def parse_engine(engine: str) -> tuple[str, str | None]:
    """'tesseract' or 'ollama:<model-tag>'; tags contain colons, so split on the first only."""
    if engine == "tesseract":
        return "tesseract", None
    kind, _, model = engine.partition(":")
    if kind != "ollama" or not model:
        raise SystemExit("extract: --engine must be 'tesseract' or 'ollama:<model-tag>'")
    return kind, model


def run_engine(
    kind: str,
    model: str | None,
    image: Path,
    schema: dict[str, Any],
    layout: dict[str, Any],
    think: bool,
) -> dict[str, Any]:
    if kind == "tesseract":
        from formbench.align import align, load_gray
        from formbench.ocr_baseline import read_form

        result = read_form(align(load_gray(image), layout), layout)
        errors = sorted(
            e.message for e in Draft202012Validator(schema).iter_errors(result["values"])
        )
        return {**result, "raw": None, "json_valid": not errors, "errors": errors}

    from formbench.vlm import extract

    return extract(image, schema, model, think=think)


def _show(value: Any) -> str:
    return "—" if value is None else str(value)


def print_report(console: Console, fields: list[dict[str, Any]], result: dict[str, Any]) -> None:
    confidence = result.get("confidence")
    table = Table(show_lines=False)
    table.add_column("Field", overflow="fold")
    table.add_column("Truth", justify="right", max_width=24)
    table.add_column("Predicted", justify="right", max_width=24)
    if confidence:
        table.add_column("Conf", justify="right")
    table.add_column("", justify="center")
    for f in fields:
        row = [f["key"], _show(f["truth"]), _show(f["predicted"])]
        if confidence:
            row.append(f"{confidence[f['key']]:.2f}")
        row.append("[green]✓[/green]" if f["correct"] else "[red]✗[/red]")
        table.add_row(*row)
    console.print(table)
    for error in result["errors"][:5]:
        console.print(f"[yellow]schema:[/yellow] {error}")


def print_metrics(console: Console, metrics: dict[str, Any]) -> None:
    table = Table(title="Metrics", show_header=False)
    for name, value in metrics.items():
        if value is None:
            shown = "n/a"
        elif name == "latency_s":
            shown = f"{value:.2f} s"
        elif name == "json_valid":
            shown = str(value)
        else:
            shown = f"{value:.1%}"
        table.add_row(name, shown)
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a photographed form and score it.")
    parser.add_argument("--form", type=Path, required=True, help="form folder from make-form")
    parser.add_argument("--image", type=Path, required=True, help="photo of the filled form")
    parser.add_argument("--engine", required=True, help="tesseract or ollama:<model-tag>")
    parser.add_argument("--think", action="store_true", help="let the VLM think (off by default)")
    parser.add_argument("--out", type=Path, default=Path("results/runs"))
    args = parser.parse_args()

    kind, model = parse_engine(args.engine)
    load_dotenv(find_dotenv(usecwd=True))
    schema = json.loads((args.form / "extract.schema.json").read_text())
    layout = json.loads((args.form / "layout.json").read_text())
    truth = json.loads((args.form / "truth.json").read_text())

    settings: dict[str, Any] = {"engine": args.engine, "model": model, "form_id": args.form.name}
    if kind == "ollama":
        settings.update(temperature=0, seed=SEED, think=args.think, prompt_version=PROMPT_VERSION)

    import openai
    from langfuse import get_client

    from formbench.align import MarkersNotFound

    langfuse = get_client()
    try:
        with langfuse.start_as_current_span(
            name="extract", input={"image": str(args.image)}, metadata=settings
        ):
            langfuse.update_current_trace(
                name=f"extract {args.form.name}", metadata=settings, tags=[kind, args.form.name]
            )
            start = time.perf_counter()
            result = run_engine(kind, model, args.image, schema, layout, args.think)
            latency = time.perf_counter() - start
            scored = score(result["values"], truth, schema, result["json_valid"])
            metrics = {**scored["metrics"], "latency_s": round(latency, 2)}
            for name, value in metrics.items():
                if value is not None:
                    langfuse.score_current_trace(name=name, value=float(value))
            langfuse.update_current_trace(output=result["values"])
            trace_id = langfuse.get_current_trace_id()
    except MarkersNotFound as error:
        raise SystemExit(f"extract: {error}") from None
    except openai.APIConnectionError:
        raise SystemExit("extract: cannot reach Ollama; is `ollama serve` running?") from None
    except openai.NotFoundError:
        raise SystemExit(f"extract: model not found; try `ollama pull {model}`") from None
    finally:
        langfuse.flush()

    console = Console()
    print_report(console, scored["fields"], result)
    print_metrics(console, metrics)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"{stamp}_{args.form.name}_{slugify(args.engine)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    run = {
        "settings": {**settings, "image": str(args.image)},
        "trace_id": trace_id,
        "metrics": metrics,
        "fields": scored["fields"],
        "confidence": result.get("confidence"),
        "raw": result["raw"],
        "errors": result["errors"],
    }
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    console.print(f"Run saved to {path}")


if __name__ == "__main__":
    main()
