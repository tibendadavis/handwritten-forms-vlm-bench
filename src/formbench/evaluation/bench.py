"""Benchmark the register pipeline on a synthetic scenario, end to end.

Each week of the scenario is read by the model, turned into a draft, reviewed by an oracle
reviewer who fixes only what the system flagged (using the truth, like a careful health
worker with the register and the child's card in hand), and approved. Scores, in layers:

- transcription: rows found or invented, cell accuracy, blanks filled or missed;
- review: blocking flags per week (reviewer workload), injected problems caught, false alarms;
- values: the approved daily values against the expected ones. Mismatches here are errors
  that reached the payload unflagged, which is what data integrity is about.
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
    _blank,
    quality_metrics,
    same_value,
    score_cover,
    score_page,
)
from formbench.files import slugify, stamp, write_json
from formbench.register.aggregate import parse_cohort, resolve_attribute_option
from formbench.register.handwriting import Unreadable
from formbench.register.spec import DOSE_COLUMNS, load_spec
from formbench.weekly.draft import names
from formbench.weekly.state import page_identity

DOSES = [c.key for c in DOSE_COLUMNS]
CHILDREN = "Children vaccinated"


def score_transcription(predicted: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    """Counts for one page photo; summed over pages before rates are taken."""
    truth_rows = {r["serial"]: r for r in truth["rows"]}
    pred_rows: dict[str, dict[str, Any]] = {}
    invented = 0
    for row in predicted["rows"]:
        serial = str(row.get("serial") or "").strip()
        if serial in truth_rows and serial not in pred_rows:
            pred_rows[serial] = row
        else:
            invented += 1
    counts = {"rows": len(truth_rows), "rows_found": len(pred_rows), "rows_invented": invented,
              "date_cells": 0, "date_correct": 0, "blank_cells": 0, "blank_filled": 0,
              "written_cells": 0, "written_missed": 0, "birth": 0, "birth_correct": 0,
              "sex_correct": 0, "card_correct": 0}  # fmt: skip
    for serial, t in truth_rows.items():
        p = pred_rows.get(serial)
        if p is None:
            continue
        counts["birth"] += not _blank(t.get("birth_date"))
        counts["birth_correct"] += not _blank(t.get("birth_date")) and same_value(t["birth_date"], p.get("birth_date"), "birth_date")  # fmt: skip
        counts["sex_correct"] += same_value(t.get("sex"), p.get("sex"), "sex")
        counts["card_correct"] += same_value(t.get("card_number"), p.get("card_number"), "card_number")  # fmt: skip
        for key in DOSES:
            tv, pv = t.get(key), p.get(key)
            if _blank(tv):
                counts["blank_cells"] += 1
                counts["blank_filled"] += not _blank(pv)
            else:
                counts["written_cells"] += 1
                counts["written_missed"] += _blank(pv)
                counts["date_cells"] += 1
                counts["date_correct"] += same_value(tv, pv, key)
    try:
        counts["cohort_correct"] = int(parse_cohort(predicted.get("cohort_month"), predicted.get("cohort_year"))
                                       == parse_cohort(truth["cohort_month"], truth["cohort_year"]))  # fmt: skip
    except Unreadable:
        counts["cohort_correct"] = 0
    return counts


def _key(v: dict[str, Any]) -> tuple[str, str, str]:
    return v["day"], v["data_element"], v["coc"]


def score_values(produced: list[dict[str, Any]], expected: list[dict[str, Any]]) -> dict[str, int]:
    got = {_key(v): v["value"] for v in produced}
    want = {_key(v): v["value"] for v in expected}
    children = {_key(v) for v in expected + produced if v["data_element_name"] == CHILDREN}
    keys = set(got) | set(want)
    return {
        "values": len(keys),
        "values_exact": sum(got.get(k) == want.get(k) for k in keys),
        "values_missing": sum(k not in got for k in want),
        "values_extra": sum(k not in want for k in got),
        "abs_error": sum(abs(got.get(k, 0) - want.get(k, 0)) for k in keys),
        "children": len(children),
        "children_exact": sum(got.get(k) == want.get(k) for k in children),
    }


def _sum(dicts: list[dict[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def _rate(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def summarize(counts: dict[str, int], weeks: int, pages: int, latency: float, valid: int) -> dict[str, Any]:  # fmt: skip
    return {
        "json_valid": _rate(valid, pages),
        "row_recall": _rate(counts["rows_found"], counts["rows"]),
        "rows_invented": counts["rows_invented"],
        "date_cell_accuracy": _rate(counts["date_correct"], counts["date_cells"]),
        "blank_hallucination_rate": _rate(counts["blank_filled"], counts["blank_cells"]),
        "written_omission_rate": _rate(counts["written_missed"], counts["written_cells"]),
        "birth_date_accuracy": _rate(counts["birth_correct"], counts["birth"]),
        "sex_accuracy": _rate(counts["sex_correct"], counts["rows_found"]),
        "card_accuracy": _rate(counts["card_correct"], counts["rows_found"]),
        "cohort_accuracy": _rate(counts["cohort_correct"], pages),
        "flags_per_week": round(counts["blocking_flags"] / weeks, 2),
        "edge_recall": _rate(counts["edges_caught"], counts["edges"]),
        "misread_flag_recall": _rate(counts["misreads_flagged"], counts["misreads"]),
        "false_alarms": counts["false_alarms"],
        "value_accuracy": _rate(counts["values_exact"], counts["values"]),
        "values_missing": counts["values_missing"],
        "values_extra": counts["values_extra"],
        "value_abs_error": counts["abs_error"],
        "children_accuracy": _rate(counts["children_exact"], counts["children"]),
        "latency_s_per_page": round(latency / max(pages, 1), 1),
    }


def score_run(weeks_out: list[dict[str, Any]], truth: dict[str, Any], ds: dict[str, Any],
              latency: float, cover: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    """All scores from what a run recorded, so saved runs can be re-scored without a model."""
    spec = load_spec(ds)
    page_counts, value_counts, review, drafts = [], [], [], []
    pages_total = valid = 0
    for week, out in zip(truth["weeks"], weeks_out):
        pages = [{"page_id": r["page_id"], **r["reading"]} for r in out["pages"]]
        for reading, photo in zip(out["pages"], week["photos"]):
            pages_total += 1
            valid += int(reading["json_valid"])
            page_counts.append(score_transcription(reading["reading"], photo["reading"]))
            page_counts.append(
                score_page(reading["reading"], photo["reading"], reading["json_valid"])
            )
        first = {"flags": out["first_flags"],
                 "unresolved_blocking": [f["id"] for f in out["first_flags"] if f["blocking"] and not f["resolved"]]}  # fmt: skip
        drafts.append(first)
        review.append(review_counts(out["first_flags"], pages, week["photos"], truth["edges"]))
        value_counts.append(score_values(out["approved_values"] + out["approved_corrections"],
                                         week["expected"]["values"] + week["expected"]["corrections"]))  # fmt: skip
    cover_expected = resolve_attribute_option(truth["cover"]["facility_type"], spec)[1] is not None
    counts = {**_sum(page_counts), **_sum(value_counts), **_sum(review),
              **flag_catch(drafts, truth["edges"], cover_expected)}  # fmt: skip
    if cover is not None:
        counts = _sum([counts, score_cover(cover, truth["cover"])])
    metrics = {**quality_metrics(counts), **summarize(counts, len(truth["weeks"]), pages_total, latency, valid)}  # fmt: skip
    return {"metrics": metrics, "counts": counts}


def read_scored_page(photo: Path, truth: dict[str, Any] | None, model: str, *, think: bool = False,
                     max_side: int | None = None) -> dict[str, Any]:  # fmt: skip
    """Read one page inside its own Langfuse span; with a truth, score it on that span."""
    from langfuse import get_client

    from formbench.register.reader import FULL_MAX_SIDE, read_page

    langfuse = get_client()
    with langfuse.start_as_current_span(name=f"page {photo.name}") as span:
        start = time.perf_counter()
        result = read_page(photo, model, think=think, max_side=max_side or FULL_MAX_SIDE)
        result["latency_s"] = round(time.perf_counter() - start, 1)
        if truth is not None:
            result["scores"] = quality_metrics(score_page(result["reading"], truth, result["json_valid"]))  # fmt: skip
            for name, value in result["scores"].items():
                if value is not None:
                    span.score(name=name, value=float(value))
            span.score(name="latency_s", value=result["latency_s"])
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
                result = read_scored_page(scenario / photo["file"], photo["reading"], model,
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


def _table(title: str, rows: list[tuple[str, dict[str, Any]]]) -> Table:
    table = Table(title=title)
    table.add_column("Metric")
    for engine, _ in rows:
        table.add_column(engine.split(":", 1)[-1][-28:], justify="right")
    for name in rows[0][1]:
        cells = []
        for _, metrics in rows:
            value = metrics[name]
            percent = isinstance(value, float) and name.endswith(
                ("accuracy", "rate", "recall", "valid")
            )
            cells.append("n/a" if value is None else (f"{value:.1%}" if percent else str(value)))
        table.add_row(name, *cells)
    return table


def combine(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per engine: counts summed over scenarios, then rates taken once (not an average of
    averages)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault(run["settings"]["engine"], []).append(run)
    out = {}
    for key, group in groups.items():
        counts = _sum([r["counts"] for r in group])
        pages = sum(len(w["pages"]) for r in group for w in r["weeks"])
        weeks = sum(len(r["weeks"]) for r in group)
        valid = sum(int(p["json_valid"]) for r in group for w in r["weeks"] for p in w["pages"])
        latency = sum(r.get("latency_s") or 0 for r in group)
        out[key] = {**quality_metrics(counts), **summarize(counts, weeks, pages, latency, valid),
                    "scenarios": len(group), "pages": pages}  # fmt: skip
    return out


def compare(args: argparse.Namespace) -> None:
    runs = [json.loads(p.read_text()) for p in args.runs]
    Console().print(_table(f"Combined over {len(runs)} run(s)", sorted(combine(runs).items())))


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
        console.print(_table(f"Register benchmark (re-scored): {args.scenario.name}", rows))
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
    console.print(_table(f"Register benchmark: {args.scenario.name}", rows))
    console.print(f"Runs saved under {args.out}/")
