"""Scores for a model's register readings against the truth.

Every score is listed once in SCORES, with a plain label, what it means and which way is
better. The terminal tables and the Langfuse score definitions both come from that list.

A cell is one value in one row and one column (or the cohort month/year at the top). A cell
is right when the model put the same value under the same column; blank on both sides is not
counted. Rows are paired by content, so one invented or missing row does not shift the rest.
Values are compared after normalizing: dates as calendar dates (12/6/26 = 12-06-26), sex by
its first letter, text ignoring case and extra spaces.

Scores are built from counts: each page adds its counts, and rates are taken once over the
sum, so a folder or several runs are never an average of averages.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any

from formbench.metrics import anls, match_lists
from formbench.register.aggregate import Key, aggregate, parse_cohort
from formbench.register.handwriting import Unreadable, parse_date
from formbench.register.prompts import ROW_FIELDS
from formbench.register.spec import DOSE_COLUMNS, RegisterSpec

DOSES = tuple(c.key for c in DOSE_COLUMNS)
DATE_KEYS = ("birth_date", *DOSES)
KEYS = tuple(ROW_FIELDS)
HEADER = ("cohort_month", "cohort_year")


@dataclass(frozen=True)
class Score:
    name: str
    label: str
    group: str  # reading | dhis2 | cost
    description: str
    higher_is_better: bool | None = True  # None: neither (a workload)
    max_value: float | None = 1.0  # None: unbounded (counts, seconds)
    percent: bool = True
    meaning: str = ""  # one plain sentence for the comment on each score in Langfuse
    formula: str = ""  # the division, in words; the comment adds the numbers


SCORES = [
    Score("overall_accuracy", "Overall accuracy", "reading",
          "THE MAIN SCORE. Of every cell that has a value on the page or in the model's output, "
          "the share with the right value under the right column.",
          meaning="How many cells were read correctly", formula="right cells ÷ cells with a value on the page or in the reading"),
    Score("completeness", "Completeness", "reading",
          "Of the entries written on the page, the share the model captured correctly. Low: it "
          "misses or misreads what is there.",
          meaning="How much of what is written was captured", formula="entries read right ÷ entries written on the page"),
    Score("reliability", "Reliability", "reading",
          "Of the values the model reported, the share that are correct. Low: it invents values "
          "or records them under the wrong column.",
          meaning="How much of what the model reported is true", formula="values right ÷ values the model reported"),
    Score("rows_detected", "Rows detected", "reading",
          "Child rows found (matched by serial number) / rows on the page.",
          meaning="How many children's rows were found", formula="rows found by serial number ÷ rows on the page"),
    Score("processing_success_rate", "Processing success rate", "reading",
          "Photos for which the model returned a usable answer (JSON in the required shape). "
          "Anything less means some photos produced nothing.",
          meaning="How many photos gave a usable answer", formula="usable answers ÷ photos"),
    Score("reporting_accuracy", "Reporting accuracy", "dhis2",
          "Daily DHIS2 values counted from the model's reading that equal those counted from the "
          "truth. Needs dose dates, birth date and sex all right, so it is stricter than cells.",
          meaning="How many daily DHIS2 numbers would be right", formula="numbers equal to the truth's ÷ all daily numbers"),
    Score("reporting_accuracy_after_review", "Reporting accuracy after review", "dhis2",
          "Synthetic scenarios only: the same, after a reviewer fixed everything that was "
          "flagged. Below 100% means wrong numbers nobody was warned about.",
          meaning="How many daily DHIS2 numbers are right after review", formula="numbers equal to the truth's ÷ all daily numbers"),
    Score("error_detection_rate", "Error detection rate", "dhis2",
          "Synthetic scenarios only: of the cells the model read wrongly, the share that raised a "
          "flag, so the reviewer got to see them.",
          meaning="How many misreads were flagged for a reviewer", formula="flagged misreads ÷ all misreads"),
    Score("review_workload", "Review workload (flags per week)", "dhis2",
          "Synthetic scenarios only: flags a reviewer must resolve per week.",
          higher_is_better=None, max_value=None, percent=False,
          meaning="How many flags a reviewer must resolve each week", formula="flags ÷ weeks"),
    Score("processing_time", "Processing time (seconds per page)", "cost",
          "Time to read one page photo on this machine.",
          higher_is_better=False, max_value=None, percent=False,
          meaning="How long one photo takes to read", formula="seconds ÷ photos"),
]  # fmt: skip
BY_NAME = {s.name: s for s in SCORES}


def _blank(value: Any) -> bool:
    return value is None or not str(value).strip()


def text(key: str, value: Any) -> str:
    """Text as compared for pairing rows: whitespace collapsed; dates as digits only."""
    if _blank(value):
        return ""
    if key in DATE_KEYS:
        return re.sub(r"\D", "", str(value))
    return " ".join(str(value).split())


def same_value(truth: Any, predicted: Any, key: str) -> bool:
    """Equal after normalization: dates by calendar date, sex by letter, text by words."""
    if _blank(truth) or _blank(predicted):
        return _blank(truth) and _blank(predicted)
    if key == "sex":
        return str(truth).strip()[:1].upper() == str(predicted).strip()[:1].upper()
    if key in DATE_KEYS:
        try:
            return parse_date(str(truth)) == parse_date(str(predicted))
        except Unreadable:
            return False
    return " ".join(str(truth).split()).casefold() == " ".join(str(predicted).split()).casefold()


def row_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Mean text similarity over the fields either row fills; used to pair rows."""
    keys = [k for k in KEYS if text(k, a.get(k)) or text(k, b.get(k))]
    if not keys:
        return 1.0
    return sum(anls(text(k, a.get(k)), text(k, b.get(k))) for k in keys) / len(keys)


def _cohort(reading: dict[str, Any]) -> str | None:
    try:
        year, month = parse_cohort(reading.get("cohort_month"), reading.get("cohort_year"))
        return f"{year}-{month:02d}"
    except Unreadable:
        return None


def daily_values(reading: dict[str, Any], spec: RegisterSpec) -> dict[Key, int]:
    """The daily DHIS2 values one page reading produces on its own, with no review: every
    dated dose counted, whatever the week (rows with a blocking problem are not counted)."""
    page = {"page_id": "page", **reading}
    return aggregate([page], spec, go_live=dt.date.min, today=dt.date.max).totals


def compare_values(got: dict[Key, int], want: dict[Key, int]) -> dict[str, int]:
    keys = set(got) | set(want)
    return {"values": len(keys), "values_exact": sum(got.get(k) == want.get(k) for k in keys)}


def cell_pairs(predicted: dict[str, Any], truth: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """(column, truth value, read value) for every cell, rows paired by content. The cohort
    month and year are included only when the truth has them."""
    pairs = [(k, truth.get(k), predicted.get(k)) for k in HEADER if k in truth]
    t_rows, p_rows = truth["rows"], predicted["rows"]
    matched = match_lists(t_rows, p_rows, row_similarity)
    seen_t, seen_p = {i for i, _, _ in matched}, {j for _, j, _ in matched}
    for i, j, _ in matched:
        pairs += [(k, t_rows[i].get(k), p_rows[j].get(k)) for k in KEYS]
    pairs += [(k, t_rows[i].get(k), None) for i in range(len(t_rows)) if i not in seen_t for k in KEYS]  # fmt: skip
    pairs += [(k, None, p_rows[j].get(k)) for j in range(len(p_rows)) if j not in seen_p for k in KEYS]  # fmt: skip
    return pairs


def score_page(
    predicted: dict[str, Any],
    truth: dict[str, Any],
    json_valid: bool,
    spec: RegisterSpec | None = None,
) -> dict[str, float]:
    """Counts for one page photo, to be summed over pages before rates are taken.

    With `spec` (from the DHIS2 metadata) the daily values the reading produces are compared
    with those the truth produces.
    """
    out: dict[str, float] = {"pages": 1, "json_valid": float(json_valid), "field_cells": 0,
                             "field_correct": 0, "field_tp": 0, "field_fp": 0, "field_fn": 0}  # fmt: skip
    for key, t, p in cell_pairs(predicted, truth):
        if _blank(t) and _blank(p):
            continue
        header = key in HEADER
        correct = _cohort(truth) == _cohort(predicted) if header else same_value(t, p, key)
        out["field_cells"] += 1
        out["field_correct"] += correct
        out["field_tp"] += correct and not _blank(p)
        out["field_fp"] += not _blank(p) and not correct
        out["field_fn"] += not _blank(t) and not correct
    serials = {str(r.get("serial") or "").strip() for r in predicted["rows"]}
    truth_serials = [str(r.get("serial") or "").strip() for r in truth["rows"]]
    out["rows"] = len(truth_serials)
    out["rows_found"] = sum(s in serials for s in truth_serials if s)
    if spec is not None:
        # A truth without the cohort is counted with the model's cohort, which is not scored.
        truth = {**{k: predicted.get(k) for k in HEADER}, **truth}
        raw = compare_values(daily_values(predicted, spec), daily_values(truth, spec))
        out["raw_values"], out["raw_values_exact"] = raw["values"], raw["values_exact"]
    return out


def sum_counts(dicts: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def _rate(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def _parts(c: dict[str, float], weeks: int | None, latency: float | None) -> dict[str, tuple[float, float]]:  # fmt: skip
    """Each score as the division it is: (numerator, denominator). A zero denominator means
    the counts cannot support the score."""
    pages, tp = c.get("pages", 0), c.get("field_tp", 0)
    return {
        "overall_accuracy": (c.get("field_correct", 0), c.get("field_cells", 0)),
        "completeness": (tp, tp + c.get("field_fn", 0)),
        "reliability": (tp, tp + c.get("field_fp", 0)),
        "rows_detected": (c.get("rows_found", 0), c.get("rows", 0)),
        "processing_success_rate": (c.get("json_valid", 0), pages),
        "reporting_accuracy": (c.get("raw_values_exact", 0), c.get("raw_values", 0)),
        "reporting_accuracy_after_review": (c.get("values_exact", 0), c.get("values", 0)),
        "error_detection_rate": (c.get("misreads_flagged", 0), c.get("misreads", 0)),
        "review_workload": (c["blocking_flags"], weeks)
        if weeks and "blocking_flags" in c
        else (0, 0),  # fmt: skip
        "processing_time": (latency, pages) if latency is not None else (0, 0),
    }


def metrics(c: dict[str, float], *, weeks: int | None = None, latency: float | None = None) -> dict[str, float | None]:  # fmt: skip
    """Scores from summed counts, in SCORES order. A score the counts cannot support (review
    scores for a photo folder, DHIS2 values without metadata) is left out."""
    places = {"review_workload": 2, "processing_time": 1}
    out = {}
    for s in SCORES:
        a, b = _parts(c, weeks, latency)[s.name]
        if b:
            out[s.name] = round(a / b, places.get(s.name, 4))
    return out


def _number(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def comments(c: dict[str, float], *, weeks: int | None = None, latency: float | None = None) -> dict[str, str]:  # fmt: skip
    """For each score in metrics(): what it means and how it was worked out, with the numbers,
    e.g. "How many cells were read correctly: right cells ÷ cells with a value ... = 190 ÷ 228"."""
    out = {}
    for s in SCORES:
        a, b = _parts(c, weeks, latency)[s.name]
        if b:
            out[s.name] = f"{s.meaning}: {s.formula} = {_number(a)} ÷ {_number(b)}"
    return out
