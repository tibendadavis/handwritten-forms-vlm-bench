"""Page-level quality scores for register transcription, in three groups.

- Text: CER and WER (corpus-level), ANLS per field, ANLSL (ANLS over the list of rows with
  optimal row matching, as in ANLS*). Dates are compared as digits only, because the same
  date is written with different separators or stacked over lines in narrow columns.
- Fields and documents: field accuracy and field F1 over cells that carry a value in the
  truth or the prediction (rows matched by content), and STP rate: share of pages with no
  error at all, which could go straight through without corrections.
- Structure: GAnTED (1 - ANLS-weighted tree edit distance / truth size, computed per matched
  row group so a page takes well under a second), Tree F1 over (row serial, field) -> value
  paths, so a value under the wrong column or child is wrong, and schema pass rate (valid JSON
  for the schema and DHIS2-compatible formats: real dates, sex M/F, cohort, numeric serial).
"""

from __future__ import annotations

import re
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
from formbench.register.aggregate import parse_cohort
from formbench.register.handwriting import Unreadable, parse_date
from formbench.register.prompts import ROW_FIELDS
from formbench.register.spec import DOSE_COLUMNS

DATE_KEYS = ("birth_date", *(c.key for c in DOSE_COLUMNS))
KEYS = tuple(ROW_FIELDS)


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


def score_page(predicted: dict[str, Any], truth: dict[str, Any], json_valid: bool) -> dict[str, float]:  # fmt: skip
    """Counts for one page, to be summed over pages before rates are taken."""
    out: dict[str, float] = dict.fromkeys(
        ("cer_edits", "cer_chars", "wer_edits", "wer_words", "anls_sum", "anls_fields",
         "anlsl_sum", "field_cells", "field_correct", "field_tp", "field_fp", "field_fn",
         "stp", "ganted_sum", "tree_tp", "tree_fp", "tree_fn", "schema_pass", "pages"), 0.0,
    )  # fmt: skip
    out["pages"] = 1
    t_rows, p_rows = truth["rows"], predicted["rows"]

    # Text: header fields, then every field of content-matched rows (unmatched rows in full).
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


def _rate(a: float, b: float) -> float | None:
    return round(a / b, 4) if b else None


def quality_metrics(c: dict[str, float]) -> dict[str, float | None]:
    """The headline quality scores from summed counts."""
    return {
        "cer": _rate(c.get("cer_edits", 0), c.get("cer_chars", 0)),
        "wer": _rate(c.get("wer_edits", 0), c.get("wer_words", 0)),
        "anls": _rate(c.get("anls_sum", 0), c.get("anls_fields", 0)),
        "anlsl": _rate(c.get("anlsl_sum", 0), c.get("pages", 0)),
        "field_accuracy": _rate(c.get("field_correct", 0), c.get("field_cells", 0)),
        "field_f1": None
        if (
            v := f1(int(c.get("field_tp", 0)), int(c.get("field_fp", 0)), int(c.get("field_fn", 0)))
        )
        is None
        else round(v, 4),  # fmt: skip
        "stp_rate": _rate(c.get("stp", 0), c.get("pages", 0)),
        "ganted": _rate(c.get("ganted_sum", 0), c.get("pages", 0)),
        "tree_f1": None
        if (v := f1(int(c.get("tree_tp", 0)), int(c.get("tree_fp", 0)), int(c.get("tree_fn", 0))))
        is None
        else round(v, 4),  # fmt: skip
        "schema_pass_rate": _rate(c.get("schema_pass", 0), c.get("pages", 0)),
        "cover_field_accuracy": _rate(c.get("cover_correct", 0), c.get("cover_fields", 0)),
    }
