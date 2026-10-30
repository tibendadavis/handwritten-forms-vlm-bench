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
from formbench.evaluation.tracking import evaluations, item_input, score_configs, sync_dataset
from formbench.files import slugify, stamp, write_json
from formbench.register.aggregate import resolve_attribute_option
from formbench.register.spec import RegisterSpec, load_spec
from formbench.weekly.draft import names
from formbench.weekly.state import page_identity

PHOTO_TYPES = (".jpg", ".jpeg", ".png")
NOT_READ = {"reading": {"cohort_month": None, "cohort_year": None, "rows": []}, "json_valid": False,
            "errors": ["not read"], "latency_s": 0.0}  # fmt: skip
GROUPS = {
    "decide": "What DHIS2 gets: rank models by these",
    "reading": "How it reads: explains the result",
    "review": "Reviewer workload",
    "cost": "Cost",
}


def _values(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], int]:
    return {(v["day"], v["data_element"], v["coc"]): v["value"] for v in rows}


def load_items(folder: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[str]]:
    """The photos of a folder with what is really written on them, as dataset items.

    A scenario folder has a truth.json (weeks, cover, expected values). Any other folder holds
    photos, each with a truth file next to it: page.jpg + page.truth.json. Returns the items,
    the scenario truth (None for a photo folder) and the photos that have no truth file.
    """
    truth_file = folder / "truth.json"
    if truth_file.exists():
        truth = json.loads(truth_file.read_text())
        items = [_item(folder, "cover.jpg", "cover", truth["cover"])]
        items += [_item(folder, p["file"], "page", p["reading"], week=w["week"])
                  for w in truth["weeks"] for p in w["photos"]]  # fmt: skip
        return items, truth, []
    items, skipped = [], []
    for photo in sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_TYPES):
        expected = photo.with_name(f"{photo.stem}.truth.json")
        if expected.exists():
            items.append(_item(folder, photo.name, "page", json.loads(expected.read_text())))
        else:
            skipped.append(photo.name)
    return items, None, skipped


def _item(folder: Path, photo: str, kind: str, expected: Any, **metadata: Any) -> dict[str, Any]:
    return {"id": f"{folder.name}/{photo}", "input": {"photo": photo, "kind": kind},
            "expected_output": expected, "metadata": {"kind": kind, **metadata}}  # fmt: skip


def read_item(folder: Path, item: dict[str, Any], model: str, *, think: bool = False,
              max_side: int | None = None) -> dict[str, Any]:  # fmt: skip
    """Read one photo (a page or the cover): the reading, its validity and the seconds taken."""
    from formbench.register.reader import FULL_MAX_SIDE, read_cover, read_page

    start = time.perf_counter()
    photo = folder / item["photo"]
    if item["kind"] == "cover":
        result = read_cover(photo, model, think=think)
    else:
        result = read_page(photo, model, think=think, max_side=max_side or FULL_MAX_SIDE)
    return {**result, "latency_s": round(time.perf_counter() - start, 1)}


def photo_scores(output: dict[str, Any], expected: Any, spec: RegisterSpec) -> dict[str, float]:
    """Scores for one photo against its truth."""
    if "cover" in output:
        return metrics(score_cover(output["cover"], expected))
    counts = score_page(output["reading"], expected, output["json_valid"], spec)
    return metrics(counts, latency=output["latency_s"])


def _values(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], int]:
    return {(v["day"], v["data_element"], v["coc"]): v["value"] for v in rows}


def review_weeks(truth: dict[str, Any], outputs: dict[str, dict[str, Any]], ds: dict[str, Any]) -> list[dict[str, Any]]:  # fmt: skip
    """Each week of a scenario: the model's readings -> draft -> oracle review -> approval."""
    spec, labels = load_spec(ds), names(ds)
    go_live = dt.date.fromisoformat(truth["go_live"])
    cover = outputs.get("cover.jpg", {}).get("cover") or {}
    state_root = Path(tempfile.mkdtemp(prefix="register-bench-"))
    weeks_out = []
    try:
        for week in truth["weeks"]:
            pages, readings = [], []
            for photo in week["photos"]:
                result = outputs.get(photo["file"]) or NOT_READ
                pid = page_identity(result["reading"], Path(photo["file"]).stem)
                pages.append({"page_id": pid, **result["reading"]})
                readings.append({"file": photo["file"], "page_id": pid,
                                 "json_valid": result["json_valid"], "reading": result["reading"]})  # fmt: skip
            first, approved = review_and_approve(
                pages, week["photos"], cover, spec, labels, week,
                go_live=go_live, intended_option=truth["intended_option"], state_root=state_root,
            )  # fmt: skip
            weeks_out.append({"week": week["week"], "pages": readings, "first_flags": first["flags"],
                              "approved_values": approved["values"], "approved_corrections": approved["corrections"]})  # fmt: skip
    finally:
        shutil.rmtree(state_root, ignore_errors=True)
    return weeks_out


def score_run(weeks_out: list[dict[str, Any]], truth: dict[str, Any], ds: dict[str, Any],
              latency: float, cover: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    """All scores of a scenario run from what it recorded, so saved runs can be re-scored
    without a model."""
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


def score_photos(pages_out: list[dict[str, Any]], items: list[dict[str, Any]], ds: dict[str, Any],
                 latency: float) -> dict[str, Any]:  # fmt: skip
    """All scores of a photo-folder run (page scores only: there are no weeks to review)."""
    spec = load_spec(ds)
    expected = {i["input"]["photo"]: i["expected_output"] for i in items}
    total = sum_counts([score_page(p["reading"], expected[p["file"]], p["json_valid"], spec)
                        for p in pages_out if p["file"] in expected])  # fmt: skip
    return {"metrics": metrics(total, latency=latency), "counts": total}


def read_scored_page(photo: Path, truth: dict[str, Any] | None, model: str, spec: RegisterSpec, *,
                     think: bool = False, max_side: int | None = None,
                     configs: dict[str, str] | None = None) -> dict[str, Any]:  # fmt: skip
    """Read one page inside its own Langfuse span; with a truth, score it on that span."""
    from langfuse import get_client

    langfuse = get_client()
    with langfuse.start_as_current_span(name=f"page {photo.name}") as span:
        item = {"photo": photo.name, "kind": "page"}
        result = read_item(photo.parent, item, model, think=think, max_side=max_side)
        if truth is not None:
            result["scores"] = photo_scores(result, truth, spec)
            for name, value in result["scores"].items():
                span.score(name=name, value=float(value), config_id=(configs or {}).get(name))
    return result


def run_bench(folder: Path, model: str, ds: dict[str, Any], *, langfuse: Any,
              configs: dict[str, str] | None, think: bool = False,
              max_side: int | None = None) -> dict[str, Any]:  # fmt: skip
    """Benchmark one model on a folder as a Langfuse experiment: one trace per photo, scored
    against its truth, then the whole-run scores on the dataset run. With configs None
    (Langfuse not reachable) the same run happens on local items."""
    items, truth, skipped = load_items(folder)
    if not items:
        raise SystemExit(f"formbench bench: no photo in {folder} has a truth file")
    spec = load_spec(ds)
    settings = {"folder": folder.name, "model": model, "think": think, "max_side": max_side}
    data = items
    if configs is not None:
        data = sync_dataset(langfuse, folder.name, items, f"Register photos and their truth: {folder}")  # fmt: skip

    def task(*, item: Any, **_: Any) -> dict[str, Any]:
        photo = item_input(item)
        langfuse.update_current_trace(name=f"{model} {photo['photo']}", tags=["formbench"], metadata=settings)  # fmt: skip
        return read_item(folder, photo, model, think=think, max_side=max_side)

    def evaluator(*, output: dict[str, Any], expected_output: Any, **_: Any) -> list[Any]:
        return evaluations(photo_scores(output, expected_output, spec), configs or {})

    run_name = " ".join([model, *(["think"] if think else []), *([f"max-side {max_side}"] if max_side else []), stamp()])  # fmt: skip
    result = langfuse.run_experiment(
        name=folder.name, run_name=run_name, data=data, task=task, evaluators=[evaluator],
        max_concurrency=1, metadata=settings,
    )  # fmt: skip
    outputs = {item_input(r.item)["photo"]: r.output for r in result.item_results}
    failed = [i["input"]["photo"] for i in items if i["input"]["photo"] not in outputs]
    if len(failed) == len(items):
        raise SystemExit("formbench bench: every photo failed. Is Ollama running (`ollama list`)?")
    latency = sum(o["latency_s"] for o in outputs.values() if "reading" in o)
    if truth is not None:
        weeks_out = review_weeks(truth, outputs, ds)
        cover = outputs.get("cover.jpg", {}).get("cover")
        record = {**score_run(weeks_out, truth, ds, latency, cover), "cover": cover, "weeks": weeks_out}  # fmt: skip
    else:
        pages_out = []
        for item in items:
            output = outputs.get(item["input"]["photo"]) or NOT_READ
            pages_out.append({"file": item["input"]["photo"], "json_valid": output["json_valid"],
                              "reading": output["reading"]})  # fmt: skip
        record = {**score_photos(pages_out, items, ds, latency), "pages": pages_out}
    if result.dataset_run_id:
        for name, value in record["metrics"].items():
            langfuse.create_score(dataset_run_id=result.dataset_run_id, name=name,
                                  value=float(value), config_id=(configs or {}).get(name))  # fmt: skip
    langfuse.flush()
    return {**record, "latency_s": round(latency, 1), "page_count": sum(i["input"]["kind"] == "page" for i in items),  # fmt: skip
            "run_name": run_name, "dataset_run_url": result.dataset_run_url, "failed": failed,
            "skipped": skipped}  # fmt: skip


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


def page_count(run: dict[str, Any]) -> int:
    return run.get("page_count") or sum(len(w["pages"]) for w in run["weeks"])


def combine(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per engine: counts summed over scenarios, then rates taken once (not an average of
    averages)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault(run["settings"]["engine"], []).append(run)
    return {
        engine: metrics(
            sum_counts([r["counts"] for r in group]),
            weeks=sum(len(r.get("weeks", [])) for r in group),
            latency=sum(r.get("latency_s") or 0 for r in group),
        )
        for engine, group in groups.items()
    }


def compare(args: argparse.Namespace) -> None:
    runs = [json.loads(p.read_text()) for p in args.runs]
    scenarios = sorted({r["settings"]["scenario"] for r in runs})
    pages = sum(page_count(r) for r in runs)
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


def _rescore(path: Path, folder: Path, ds: dict[str, Any]) -> dict[str, Any]:
    run = json.loads(path.read_text())
    items, truth, _ = load_items(folder)
    pages = page_count(run)
    latency = run.get("latency_s") or run["metrics"]["latency_s_per_page"] * pages
    if truth is not None:
        run.update(score_run(run["weeks"], truth, ds, latency, run.get("cover")))
    else:
        run.update(score_photos(run["pages"], items, ds, latency))
    write_json(path, run)
    return run


def command(args: argparse.Namespace) -> None:
    from langfuse import get_client

    from formbench.cli import model_tag

    ds = json.loads(args.metadata.read_text())
    folder = args.scenario
    console = Console()
    rows = []
    if args.rescore:
        for path in args.rescore:
            run = _rescore(path, folder, ds)
            rows.append((run["settings"]["engine"], run["metrics"]))
        console.print(score_table(f"Re-scored: {folder.name}", rows))
        return
    if not args.engine:
        raise SystemExit("formbench bench: give --engine (or --rescore saved run files)")

    langfuse = get_client()
    try:
        configs: dict[str, str] | None = score_configs(langfuse)
    except Exception as error:  # no keys, or the server is down: still run, save locally
        console.print(f"[yellow]Langfuse not reachable ({error}); results are saved locally only.[/yellow]")  # fmt: skip
        configs = None
    for engine in args.engine:
        model = model_tag(engine)
        console.print(f"Running {engine} on {folder.name}...")
        result = run_bench(folder, model, ds, langfuse=langfuse, configs=configs,
                           think=args.think, max_side=args.max_side)  # fmt: skip
        settings = {"scenario": folder.name, "engine": engine, "think": args.think, "max_side": args.max_side}  # fmt: skip
        path = write_json(args.out / f"{stamp()}_{folder.name}_{slugify(engine)}.json", {"settings": settings, **result})  # fmt: skip
        for note, photos in (("not read (failed)", result["failed"]), ("skipped (no truth file)", result["skipped"])):  # fmt: skip
            if photos:
                console.print(
                    f"[yellow]{len(photos)} photo(s) {note}: {', '.join(photos)}[/yellow]"
                )
        if result["dataset_run_url"]:
            console.print(f"Langfuse: {result['dataset_run_url']}")
        console.print(f"Saved to {path}")
        rows.append((engine, result["metrics"]))
    console.print(score_table(f"Register benchmark: {folder.name}", rows))
