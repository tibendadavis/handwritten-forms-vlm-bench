"""Scores for a model's register readings against the truth.

Every score is listed once in SCORES, with what it means, its range and which way is better.
The terminal tables and the Langfuse score definitions both come from that list.

Scores are built from counts: each page adds its counts, and rates are taken once over the
sum, so a scenario or a whole benchmark is never an average of averages.

How values are compared: whitespace is collapsed; dates are compared as calendar dates for
cells and as digits for text scores (12/6/26 and 12-06-26 are the same thing, not an error);
truth and predicted rows are paired by content, so one invented row does not shift the rest.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any

from formbench.metrics import (
    anls,
    anls_list,
    char_edits,
    f1,
    match_lists,
    to_tree,
    tree_distance,
    word_edits,
)
from formbench.register.aggregate import Key, aggregate, parse_cohort
from formbench.register.handwriting import Unreadable, parse_date
from formbench.register.prompts import ROW_FIELDS
from formbench.register.spec import DOSE_COLUMNS, RegisterSpec

DOSES = tuple(c.key for c in DOSE_COLUMNS)
DATE_KEYS = ("birth_date", *DOSES)
KEYS = tuple(ROW_FIELDS)


@dataclass(frozen=True)
class Score:
    name: str
    group: str  # decide | reading | review | cost
    description: str
    higher_is_better: bool | None = True  # None: neither (a workload)
    max_value: float | None = 1.0  # None: unbounded (counts, seconds)
    percent: bool = True


SCORES = [
    # What DHIS2 ends up with: rank models by these.
    Score(
        "value_accuracy_after_review",
        "decide",
        "Daily DHIS2 values (vaccine, dose, age band, sex, day) exactly right after the reviewer fixed everything that was flagged. Anything below 100% is a wrong number nobody was warned about.",
    ),  # fmt: skip
    Score(
        "value_accuracy_before_review",
        "decide",
        "Daily DHIS2 values counted straight from the model's reading that exactly match the values counted from the truth, before any review.",
    ),  # fmt: skip
    Score(
        "misread_flag_recall",
        "decide",
        "Of the cells the model read wrongly, the share that raised a flag, so the reviewer got to see them.",
    ),  # fmt: skip
    Score(
        "field_f1",
        "decide",
        "Cells right, balancing 'what the model wrote is right' (no invented values) and 'what is on the page was found' (no missed values).",
        percent=False,
    ),  # fmt: skip
    Score(
        "tree_f1",
        "decide",
        "Like field F1, but a value only counts under the right row serial and the right vaccine column: catches values filed under the wrong header.",
        percent=False,
    ),  # fmt: skip
    # How it reads: explain the result.
    Score(
        "cer",
        "reading",
        "Character error rate: wrong, missing or extra characters / characters in the truth. 0 is perfect.",
        higher_is_better=False,
        max_value=None,
        percent=False,
    ),  # fmt: skip
    Score(
        "wer",
        "reading",
        "Word error rate: the same in whole words (a date is one word). 0 is perfect.",
        higher_is_better=False,
        max_value=None,
        percent=False,
    ),  # fmt: skip
    Score(
        "anls",
        "reading",
        "Average text similarity per cell (1 - normalized edit distance; under 0.5 counts as 0). Partial credit, so a nearly right date still scores.",
        percent=False,
    ),  # fmt: skip
    Score(
        "anlsl",
        "reading",
        "ANLS for the list of rows on a page: rows paired by content, and missing or invented rows score 0.",
        percent=False,
    ),  # fmt: skip
    Score(
        "field_accuracy",
        "reading",
        "Cells read exactly right / cells with a value in the truth or the reading (blank on both sides is not counted).",
    ),  # fmt: skip
    Score(
        "ganted",
        "reading",
        "Structure similarity: 1 - tree edit distance (rows, fields, values; nearly right values cost less) / truth size. Gentler than Tree F1.",
        percent=False,
    ),  # fmt: skip
    Score(
        "stp_rate",
        "reading",
        "Straight-through processing: share of page photos read with no error at all.",
    ),  # fmt: skip
    Score(
        "schema_pass_rate",
        "reading",
        "Share of page photos whose output is valid JSON with DHIS2-usable formats: real dates, sex M or F, a readable cohort, numeric serials.",
    ),  # fmt: skip
    Score(
        "json_valid",
        "reading",
        "Share of page photos where the model returned JSON that matches the schema.",
    ),  # fmt: skip
    Score(
        "row_recall", "reading", "Child rows found (by serial number) / rows in the truth."
    ),  # fmt: skip
    Score(
        "rows_invented",
        "reading",
        "Rows the model returned that are not on the page.",
        higher_is_better=False,
        max_value=None,
        percent=False,
    ),  # fmt: skip
    Score(
        "omission_rate",
        "reading",
        "Written dose dates the model left empty / written dose dates.",
        higher_is_better=False,
    ),  # fmt: skip
    Score(
        "hallucination_rate",
        "reading",
        "Empty dose cells the model filled with a date / empty dose cells.",
        higher_is_better=False,
    ),  # fmt: skip
    Score(
        "cover_field_accuracy",
        "reading",
        "Cover fields read right: facility name, MFL code, facility type, district, region.",
    ),  # fmt: skip
    # What the reviewer has to do.
    Score(
        "flags_per_week",
        "review",
        "Blocking flags a reviewer must resolve per week. Lower is not automatically better: a skipped date raises no flag.",
        higher_is_better=None,
        max_value=None,
        percent=False,
    ),  # fmt: skip
    Score(
        "edge_recall",
        "review",
        "Problems planted in the synthetic register (missing birth date, wrong cohort page, future date, unclear facility type) that were flagged.",
    ),  # fmt: skip
    Score(
        "false_alarms",
        "review",
        "Blocking flags raised on cells that were read right.",
        higher_is_better=False,
        max_value=None,
        percent=False,
    ),  # fmt: skip
    Score(
        "values_missing",
        "review",
        "Expected daily DHIS2 values absent after review.",
        higher_is_better=False,
        max_value=None,
        percent=False,
    ),  # fmt: skip
    Score(
        "values_extra",
        "review",
        "Daily DHIS2 values produced after review that should not exist.",
        higher_is_better=False,
        max_value=None,
        percent=False,
    ),  # fmt: skip
    # What it costs.
    Score(
        "latency_s_per_page",
        "cost",
        "Seconds per page photo on this machine.",
        higher_is_better=False,
        max_value=None,
        percent=False,
    ),  # fmt: skip
]
BY_NAME = {s.name: s for s in SCORES}


def _blank(value: Any) -> bool:
    return value is None or not str(value).strip()


def text(key: str, value: Any) -> str:
    """Text as compared for CER/WER/ANLS: whitespace collapsed; dates as digits only."""
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


def _canonical(key: str, value: Any) -> str | None:
    if _blank(value):
        return None
    if key in DATE_KEYS:
        try:
            return parse_date(str(value)).isoformat()
        except Unreadable:
            return f"?{text(key, value)}"
    if key == "sex":
        return str(value).strip()[:1].upper()
    return " ".join(str(value).split()).casefold()


def row_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Mean ANLS over the fields either row fills; used to pair rows."""
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


def schema_pass(reading: dict[str, Any], json_valid: bool) -> bool:
    """Valid against the JSON Schema and every value in a DHIS2-usable format."""
    if not json_valid or _cohort(reading) is None:
        return False
    for row in reading["rows"]:
        if not _blank(row.get("serial")) and not str(row["serial"]).strip().isdigit():
            return False
        if not _blank(row.get("sex")) and str(row["sex"]).strip().upper()[:1] not in ("F", "M"):
            return False
        for key in DATE_KEYS:
            try:
                parse_date(row.get(key))
            except Unreadable:
                return False
    return True


def _tree(row: dict[str, Any]) -> dict[str, Any]:
    return {k: text(k, row.get(k)) for k in KEYS}


def _tree_size(row: dict[str, Any]) -> int:
    node = to_tree(_tree(row))
    stack, size = [node], 0
    while stack:
        n = stack.pop()
        size += 1
        stack.extend(n.children)
    return size - 1


def daily_values(reading: dict[str, Any], spec: RegisterSpec) -> dict[Key, int]:
    """The daily DHIS2 values one page reading produces on its own, with no review: every
    dated dose counted, whatever the week (rows with a blocking problem are not counted)."""
    page = {"page_id": "page", **reading}
    return aggregate([page], spec, go_live=dt.date.min, today=dt.date.max).totals


def compare_values(got: dict[Key, int], want: dict[Key, int]) -> dict[str, int]:
    keys = set(got) | set(want)
    return {
        "values": len(keys),
        "values_exact": sum(got.get(k) == want.get(k) for k in keys),
        "values_missing": sum(k not in got for k in want),
        "values_extra": sum(k not in want for k in got),
    }


def _row_counts(predicted: dict[str, Any], truth: dict[str, Any]) -> dict[str, float]:
    """Rows found and dose cells missed or made up; rows matched by serial number."""
    truth_rows = {r["serial"]: r for r in truth["rows"]}
    pred_rows: dict[str, dict[str, Any]] = {}
    invented = 0
    for row in predicted["rows"]:
        serial = str(row.get("serial") or "").strip()
        if serial in truth_rows and serial not in pred_rows:
            pred_rows[serial] = row
        else:
            invented += 1
    out = {"rows": len(truth_rows), "rows_found": len(pred_rows), "rows_invented": invented,
           "written_cells": 0, "written_missed": 0, "blank_cells": 0, "blank_filled": 0}  # fmt: skip
    for serial, t in truth_rows.items():
        p = pred_rows.get(serial, {})
        for key in DOSES:
            if _blank(t.get(key)):
                out["blank_cells"] += serial in pred_rows
                out["blank_filled"] += serial in pred_rows and not _blank(p.get(key))
            else:
                out["written_cells"] += 1
                out["written_missed"] += _blank(p.get(key))
    return out


def score_page(
    predicted: dict[str, Any],
    truth: dict[str, Any],
    json_valid: bool,
    spec: RegisterSpec | None = None,
) -> dict[str, float]:
    """Counts for one page photo, to be summed over pages before rates are taken.

    With `spec` (from the DHIS2 metadata) the daily values the reading produces are compared
    with those the truth produces: the score that matters for aggregate reporting.
    """
    out: dict[str, float] = dict.fromkeys(
        ("cer_edits", "cer_chars", "wer_edits", "wer_words", "anls_sum", "anls_fields",
         "anlsl_sum", "field_cells", "field_correct", "field_tp", "field_fp", "field_fn",
         "stp", "ganted_sum", "tree_tp", "tree_fp", "tree_fn", "schema_pass"), 0.0,
    )  # fmt: skip
    out.update(pages=1, json_valid=float(json_valid), **_row_counts(predicted, truth))
    t_rows, p_rows = truth["rows"], predicted["rows"]

    # Text and cells: header fields, then every field of content-matched rows (unmatched in full).
    pairs = [("cohort_month", truth.get("cohort_month"), predicted.get("cohort_month")),
             ("cohort_year", truth.get("cohort_year"), predicted.get("cohort_year"))]  # fmt: skip
    matched = match_lists(t_rows, p_rows, row_similarity)
    seen_t, seen_p = {i for i, _, _ in matched}, {j for _, j, _ in matched}
    for i, j, _ in matched:
        pairs += [(k, t_rows[i].get(k), p_rows[j].get(k)) for k in KEYS]
    pairs += [(k, t_rows[i].get(k), None) for i in range(len(t_rows)) if i not in seen_t for k in KEYS]  # fmt: skip
    pairs += [(k, None, p_rows[j].get(k)) for j in range(len(p_rows)) if j not in seen_p for k in KEYS]  # fmt: skip
    for key, t, p in pairs:
        tt, pt = text(key, t), text(key, p)
        if tt:
            edits, chars = char_edits(tt, pt)
            out["cer_edits"] += edits
            out["cer_chars"] += chars
            edits, words = word_edits(tt, pt)
            out["wer_edits"] += edits
            out["wer_words"] += words
        if tt or pt:
            out["anls_sum"] += anls(tt, pt)
            out["anls_fields"] += 1
            correct = (
                _cohort(truth) == _cohort(predicted)
                if key.startswith("cohort_")
                else same_value(t, p, key)
            )
            out["field_cells"] += 1
            out["field_correct"] += correct
            out["field_tp"] += correct and bool(pt)
            out["field_fp"] += bool(pt) and not correct
            out["field_fn"] += bool(tt) and not correct
    out["anlsl_sum"] = anls_list(t_rows, p_rows, row_similarity)
    errors = out["field_fp"] + out["field_fn"]
    out["stp"] = float(errors == 0 and json_valid and len(t_rows) == len(p_rows))

    # Structure: GAnTED per matched row group; unmatched rows cost their whole size.
    cost = sum(tree_distance(to_tree(_tree(t_rows[i])), to_tree(_tree(p_rows[j]))) for i, j, _ in matched)  # fmt: skip
    cost += sum(_tree_size(t_rows[i]) for i in range(len(t_rows)) if i not in seen_t)
    cost += sum(_tree_size(p_rows[j]) for j in range(len(p_rows)) if j not in seen_p)
    header_ok = _cohort(truth) == _cohort(predicted)
    size = sum(_tree_size(r) for r in t_rows) + 2
    out["ganted_sum"] = max(0.0, 1 - (cost + (0 if header_ok else 2)) / size)

    def paths(reading: dict[str, Any]) -> set[tuple[str, ...]]:
        out_paths = {("cohort", _cohort(reading) or "?")}
        for row in reading["rows"]:
            serial = text("serial", row.get("serial"))
            for key in KEYS:
                value = _canonical(key, row.get(key))
                if value is not None:
                    out_paths.add(("rows", serial, key, value))
        return out_paths

    tp_paths, pp_paths = paths(truth), paths(predicted)
    out["tree_tp"] = len(tp_paths & pp_paths)
    out["tree_fp"] = len(pp_paths - tp_paths)
    out["tree_fn"] = len(tp_paths - pp_paths)
    out["schema_pass"] = float(schema_pass(predicted, json_valid))
    if spec is not None:
        raw = compare_values(daily_values(predicted, spec), daily_values(truth, spec))
        out["raw_values"], out["raw_values_exact"] = raw["values"], raw["values_exact"]
    return out


def score_cover(predicted: dict[str, Any], truth: dict[str, Any]) -> dict[str, float]:
    """Text and field counts for the register cover (facility name, MFL code, type, ...)."""
    out = {"cer_edits": 0.0, "cer_chars": 0.0, "wer_edits": 0.0, "wer_words": 0.0,
           "anls_sum": 0.0, "anls_fields": 0.0, "cover_fields": 0.0, "cover_correct": 0.0}  # fmt: skip
    for key in ("facility_name", "mfl_code", "facility_type", "district", "region"):
        tt, pt = text(key, truth.get(key)), text(key, predicted.get(key))
        if tt:
            edits, chars = char_edits(tt, pt)
            out["cer_edits"] += edits
            out["cer_chars"] += chars
            edits, words = word_edits(tt, pt)
            out["wer_edits"] += edits
            out["wer_words"] += words
        if tt or pt:
            out["anls_sum"] += anls(tt, pt)
            out["anls_fields"] += 1
            out["cover_fields"] += 1
            out["cover_correct"] += same_value(tt, pt, key)
    return out


def sum_counts(dicts: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def _rate(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def _f1(c: dict[str, float], prefix: str) -> float | None:
    value = f1(int(c.get(f"{prefix}_tp", 0)), int(c.get(f"{prefix}_fp", 0)), int(c.get(f"{prefix}_fn", 0)))  # fmt: skip
    return None if value is None else round(value, 4)


def metrics(c: dict[str, float], *, weeks: int | None = None, latency: float | None = None) -> dict[str, float | None]:  # fmt: skip
    """Scores from summed counts, in SCORES order. A score that needs data the counts do not
    have (review scores for a single photo, cover scores without a cover) is left out."""
    pages = c.get("pages", 0)
    values = {
        "value_accuracy_after_review": _rate(c.get("values_exact", 0), c.get("values", 0)),
        "value_accuracy_before_review": _rate(c.get("raw_values_exact", 0), c.get("raw_values", 0)),
        "misread_flag_recall": _rate(c.get("misreads_flagged", 0), c.get("misreads", 0)),
        "field_f1": _f1(c, "field"),
        "tree_f1": _f1(c, "tree"),
        "cer": _rate(c.get("cer_edits", 0), c.get("cer_chars", 0)),
        "wer": _rate(c.get("wer_edits", 0), c.get("wer_words", 0)),
        "anls": _rate(c.get("anls_sum", 0), c.get("anls_fields", 0)),
        "anlsl": _rate(c.get("anlsl_sum", 0), pages),
        "field_accuracy": _rate(c.get("field_correct", 0), c.get("field_cells", 0)),
        "ganted": _rate(c.get("ganted_sum", 0), pages),
        "stp_rate": _rate(c.get("stp", 0), pages),
        "schema_pass_rate": _rate(c.get("schema_pass", 0), pages),
        "json_valid": _rate(c.get("json_valid", 0), pages),
        "row_recall": _rate(c.get("rows_found", 0), c.get("rows", 0)),
        "rows_invented": c.get("rows_invented"),
        "omission_rate": _rate(c.get("written_missed", 0), c.get("written_cells", 0)),
        "hallucination_rate": _rate(c.get("blank_filled", 0), c.get("blank_cells", 0)),
        "cover_field_accuracy": _rate(c.get("cover_correct", 0), c.get("cover_fields", 0)),
        "flags_per_week": round(c["blocking_flags"] / weeks, 2)
        if weeks and "blocking_flags" in c
        else None,  # fmt: skip
        "edge_recall": _rate(c.get("edges_caught", 0), c.get("edges", 0)),
        "false_alarms": c.get("false_alarms"),
        "values_missing": c.get("values_missing"),
        "values_extra": c.get("values_extra"),
        "latency_s_per_page": round(latency / pages, 1) if latency is not None and pages else None,
    }
    for name in ("rows_invented", "false_alarms", "values_missing", "values_extra"):
        if values[name] is not None:
            values[name] = int(values[name])
    return {s.name: values[s.name] for s in SCORES if values[s.name] is not None}
