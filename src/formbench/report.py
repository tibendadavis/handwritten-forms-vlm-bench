"""Terminal tables: a run's scores, runs compared, one run broken down, and what scores mean."""

from __future__ import annotations

from collections import Counter
from typing import Any

from rich.table import Table

from formbench.runner import run_label
from formbench.scoring import BY_NAME, SCORES, Score, metrics, sum_counts

NOT_COMPUTED = "–"
RESULTS = ("right", "misread", "missed", "invented")


def show(score: Score, value: float | None) -> str:
    """A score as shown in tables: rates as a percentage with one decimal, seconds as is."""
    if value is None:
        return NOT_COMPUTED
    return f"{value:.1%}" if score.is_rate else f"{value:.1f}"


def score_table(title: str, columns: list[tuple[str, dict[str, float]]]) -> Table:
    """A row per score (with the way that is better), a column per run. Scores no run has are
    left out."""
    table = Table(title=title, caption="↑ higher is better, ↓ lower is better (formbench scores)")
    table.add_column("Score")
    for label, _ in columns:
        table.add_column(label, justify="right")
    for score in SCORES:
        if any(score.name in m for _, m in columns):
            table.add_row(score.heading, *(show(score, m.get(score.name)) for _, m in columns))
    return table


def pool(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Runs with the same label pooled: counts and seconds summed, then scores taken once."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault(run_label(run["settings"]), []).append(run)
    return {
        label: metrics(
            sum_counts(r["counts"] for r in group),
            sum(r.get("latency_s") or 0 for r in group),
        )
        for label, group in sorted(groups.items())
    }


def compare_tables(runs: list[dict[str, Any]]) -> list[Table]:
    """One table per folder, a column per run label."""
    tables = []
    for folder in sorted({r["settings"]["folder"] for r in runs}):
        group = [r for r in runs if r["settings"]["folder"] == folder]
        photos = max(r.get("photo_count") or 0 for r in group)
        title = f"{folder}: {photos} photo(s), {len(group)} run(s)"
        tables.append(score_table(title, list(pool(group).items())))
    return tables


def breakdown_table(run: dict[str, Any], by: str = "group") -> Table:
    """Accuracy per group (section or register column) or per field type, so it shows what a
    model reads well and what it does not."""
    tallies: dict[str, Counter] = {}
    for cell in run["outcomes"]:
        tallies.setdefault(str(cell[by]), Counter())[cell["result"]] += 1
    title = f"{run_label(run['settings'])} on {run['settings']['folder']}, by {by}"
    table = Table(title=title)
    table.add_column("Group" if by == "group" else "Type")
    for heading in ("Cells", "Right", "Misread", "Missed", "Invented", "Accuracy"):
        table.add_column(heading, justify="right")
    accuracy = BY_NAME["overall_accuracy"]

    def add(name: str, tally: Counter, **style: Any) -> None:
        cells = sum(tally.values())
        rate = round(tally["right"] / cells, 4) if cells else None
        counts = [str(tally[r]) for r in RESULTS]
        table.add_row(name, str(cells), *counts, show(accuracy, rate), **style)

    for name, tally in tallies.items():
        add(name, tally)
    table.add_section()
    add("All", sum(tallies.values(), Counter()), style="bold")
    return table


def scores_table() -> Table:
    """Every score: its label, its name in Langfuse and what it means."""
    table = Table(title="Scores", caption="↑ higher is better, ↓ lower is better")
    table.add_column("Score")
    table.add_column("Name in Langfuse")
    table.add_column("What it means")
    for score in SCORES:
        table.add_row(score.heading, score.name, score.description)
    return table


def check_table(items: list[dict[str, Any]]) -> Table:
    """The photos a run would read, grouped by sheet (cv-form) or listed (register)."""
    table = Table(title=f"{len(items)} photo(s) to read")
    table.add_column("Sheet or photo")
    table.add_column("Photos", justify="right")
    table.add_column("Parts")
    groups: dict[str, list[str]] = {}
    for item in items:
        meta, photo = item["metadata"], item["input"]
        sheet = meta.get("sheet") or photo["photo"]
        if "page" in photo:
            part = f"page {photo['page']} ({photo['sections']})"
        else:
            part = photo.get("section") or "page"
        groups.setdefault(sheet, []).append(part)
    for sheet, parts in groups.items():
        table.add_row(sheet, str(len(parts)), ", ".join(parts))
    return table
