"""Benchmark models on a scenario (photos + truth.json), end to end.

Each week of the scenario is read by the model, turned into a draft, reviewed by the oracle
reviewer (evaluation/review.py) who fixes only what the system flagged, and approved. The
approved daily values are compared with the expected ones: a mismatch there is an error that
reached the DHIS2 payload unflagged, which is what data integrity is about.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from formbench.evaluation.review import flag_catch, review_and_approve, review_counts
from formbench.evaluation.scoring import (
    SCORES,
    Score,
    compare_values,
    metrics,
    score_cover,
    score_page,
    sum_counts,
)
from formbench.files import slugify, stamp, write_json
from formbench.register.aggregate import resolve_attribute_option
from formbench.register.spec import RegisterSpec, load_spec
from formbench.weekly.draft import names
from formbench.weekly.state import page_identity

GROUPS = {
    "decide": "What DHIS2 gets: rank models by these",
    "reading": "How it reads: explains the result",
    "review": "Reviewer workload",
    "cost": "Cost",
}


def _values(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], int]:
    return {(v["day"], v["data_element"], v["coc"]): v["value"] for v in rows}


def score_run(weeks_out: list[dict[str, Any]], truth: dict[str, Any], ds: dict[str, Any],
              latency: float, cover: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    """All scores from what a run recorded, so saved runs can be re-scored without a model."""
    spec = load_spec(ds)
    counts, drafts = [], []
    for week, out in zip(truth["weeks"], weeks_out):
        pages = [{"page_id": r["page_id"], **r["reading"]} for r in out["pages"]]
        for reading, photo in zip(out["pages"], week["photos"]):
            counts.append(score_page(reading["reading"], photo["reading"], reading["json_valid"], spec))  # fmt: skip
        drafts.append({"flags": out["first_flags"],
                       "unresolved_blocking": [f["id"] for f in out["first_flags"] if f["blocking"] and not f["resolved"]]})  # fmt: skip
        counts.append(review_counts(out["first_flags"], pages, week["photos"], truth["edges"]))
        counts.append(compare_values(_values(out["approved_values"] + out["approved_corrections"]),
                                     _values(week["expected"]["values"] + week["expected"]["corrections"])))  # fmt: skip
    cover_expected = resolve_attribute_option(truth["cover"]["facility_type"], spec)[1] is not None
    counts.append(flag_catch(drafts, truth["edges"], cover_expected))
    if cover is not None:
        counts.append(score_cover(cover, truth["cover"]))
    total = sum_counts(counts)
    return {"metrics": metrics(total, weeks=len(truth["weeks"]), latency=latency), "counts": total}


def read_scored_page(photo: Path, truth: dict[str, Any] | None, model: str, spec: RegisterSpec, *,
                     think: bool = False, max_side: int | None = None) -> dict[str, Any]:  # fmt: skip
    """Read one page inside its own Langfuse span; with a truth, score it on that span."""
    from langfuse import get_client

    from formbench.register.reader import FULL_MAX_SIDE, read_page

    langfuse = get_client()
    with langfuse.start_as_current_span(name=f"page {photo.name}") as span:
        start = time.perf_counter()
        result = read_page(photo, model, think=think, max_side=max_side or FULL_MAX_SIDE)
        result["latency_s"] = round(time.perf_counter() - start, 1)
        if truth is not None:
            counts = score_page(result["reading"], truth, result["json_valid"], spec)
            result["scores"] = metrics(counts, latency=result["latency_s"])
            for name, value in result["scores"].items():
                span.score(name=name, value=float(value))
    return result


def run_bench(scenario: Path, model: str, ds: dict[str, Any], think: bool = False,
              max_side: int | None = None) -> dict[str, Any]:  # fmt: skip
    from formbench.register.reader import read_cover

    truth = json.loads((scenario / "truth.json").read_text())
    spec, labels = load_spec(ds), names(ds)
    go_live = dt.date.fromisoformat(truth["go_live"])
    state_root = Path(tempfile.mkdtemp(prefix="register-bench-"))
    start = time.perf_counter()
    weeks_out = []
    try:
        cover = read_cover(scenario / "cover.jpg", model, think=think)
        for week in truth["weeks"]:
            pages, readings = [], []
            for photo in week["photos"]:
                result = read_scored_page(scenario / photo["file"], photo["reading"], model, spec,
                                          think=think, max_side=max_side)  # fmt: skip
                pid = page_identity(result["reading"], Path(photo["file"]).stem)
                pages.append({"page_id": pid, **result["reading"]})
                readings.append({"file": photo["file"], "page_id": pid,
                                 "json_valid": result["json_valid"], "reading": result["reading"]})  # fmt: skip
            first, approved = review_and_approve(
                pages, week["photos"], cover["cover"], spec, labels, week,
                go_live=go_live, intended_option=truth["intended_option"], state_root=state_root,
            )  # fmt: skip
            weeks_out.append({"week": week["week"], "pages": readings, "first_flags": first["flags"],
                              "approved_values": approved["values"], "approved_corrections": approved["corrections"]})  # fmt: skip
        latency = time.perf_counter() - start
    finally:
        shutil.rmtree(state_root, ignore_errors=True)
    scored = score_run(weeks_out, truth, ds, latency, cover["cover"])
    return {**scored, "latency_s": round(latency, 1), "cover": cover["cover"], "weeks": weeks_out}


def _cell(score: Score, value: Any) -> str:
    if value is None:
        return "–"
    if score.percent:
        return f"{value:.1%}"
    return str(value) if isinstance(value, int) else f"{value:.4g}"


def score_table(title: str, columns: list[tuple[str, dict[str, Any]]]) -> Table:
    """Scores grouped by what they are for; ↑ higher is better, ↓ lower is better."""
    table = Table(title=title)
    table.add_column("Score")
    for label, _ in columns:
        table.add_column(label.split(":", 1)[-1][-28:], justify="right")
    for group, heading in GROUPS.items():
        scores = [s for s in SCORES if s.group == group and any(s.name in m for _, m in columns)]
        if not scores:
            continue
        table.add_row(f"[bold]{heading}[/bold]", end_section=False)
        for s in scores:
            arrow = {True: " ↑", False: " ↓"}.get(s.higher_is_better, "")
            table.add_row(f"  {s.name}{arrow}", *(_cell(s, m.get(s.name)) for _, m in columns))
        table.add_section()
    return table


def combine(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per engine: counts summed over scenarios, then rates taken once (not an average of
    averages)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault(run["settings"]["engine"], []).append(run)
    return {
        engine: metrics(
            sum_counts([r["counts"] for r in group]),
            weeks=sum(len(r["weeks"]) for r in group),
            latency=sum(r.get("latency_s") or 0 for r in group),
        )
        for engine, group in groups.items()
    }


def compare(args: argparse.Namespace) -> None:
    runs = [json.loads(p.read_text()) for p in args.runs]
    scenarios = sorted({r["settings"]["scenario"] for r in runs})
    pages = sum(len(w["pages"]) for r in runs for w in r["weeks"])
    title = f"{len(scenarios)} scenario(s), {pages} page photos: {', '.join(scenarios)}"
    Console().print(score_table(title, sorted(combine(runs).items())))


def explain(args: argparse.Namespace) -> None:
    """Print what every score means, grouped as in the benchmark tables."""
    table = Table(title="Scores (↑ higher is better, ↓ lower is better)", show_lines=False)
    table.add_column("Score")
    table.add_column("What it means")
    for group, heading in GROUPS.items():
        table.add_row(f"[bold]{heading}[/bold]", "")
        for s in (s for s in SCORES if s.group == group):
            arrow = {True: " ↑", False: " ↓"}.get(s.higher_is_better, "")
            table.add_row(f"  {s.name}{arrow}", s.description)
        table.add_section()
    Console().print(table)


def command(args: argparse.Namespace) -> None:
    from formbench.cli import model_tag

    ds = json.loads(args.metadata.read_text())
    console = Console()
    rows = []
    if args.rescore:
        truth = json.loads((args.scenario / "truth.json").read_text())
        for path in args.rescore:
            run = json.loads(path.read_text())
            latency = run.get("latency_s") or run["metrics"]["latency_s_per_page"] * sum(len(w["pages"]) for w in run["weeks"])  # fmt: skip
            run.update(
                score_run(run["weeks"], truth, ds, latency, run.get("cover")), latency_s=latency
            )
            write_json(path, run)
            rows.append((run["settings"]["engine"], run["metrics"]))
        console.print(score_table(f"Register benchmark (re-scored): {args.scenario.name}", rows))
        return
    if not args.engine:
        raise SystemExit("formbench bench: give --engine (or --rescore saved run files)")

    from langfuse import get_client

    langfuse = get_client()
    for engine in args.engine:
        model = model_tag(engine)
        settings = {"scenario": args.scenario.name, "engine": engine, "think": args.think,
                    "max_side": args.max_side}  # fmt: skip
        console.print(f"Running {engine} on {args.scenario.name}...")
        try:
            with langfuse.start_as_current_span(name="register-bench", metadata=settings):
                langfuse.update_current_trace(name=f"register-bench {args.scenario.name}", metadata=settings, tags=["register-bench"])  # fmt: skip
                result = run_bench(
                    args.scenario, model, ds, think=args.think, max_side=args.max_side
                )
                for name, value in result["metrics"].items():
                    if value is not None:
                        langfuse.score_current_trace(name=name, value=float(value))
                trace_id = langfuse.get_current_trace_id()
        finally:
            langfuse.flush()
        path = args.out / f"{stamp()}_{args.scenario.name}_{slugify(engine)}.json"
        write_json(path, {"settings": settings, "trace_id": trace_id, **result})
        rows.append((engine, result["metrics"]))
    console.print(score_table(f"Register benchmark: {args.scenario.name}", rows))
    console.print(f"Runs saved under {args.out}/")
