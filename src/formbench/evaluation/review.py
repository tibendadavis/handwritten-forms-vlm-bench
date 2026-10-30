"""The benchmark's stand-in reviewer and the integrity scores it measures.

A real person resolves every blocking flag before approval. The oracle reviewer does the
same with the truth in hand, and fixes only what was flagged, so any error left in the
approved values is one nobody was warned about.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from formbench.evaluation.scoring import _blank, same_value
from formbench.register.aggregate import parse_cohort
from formbench.register.spec import DOSE_COLUMNS
from formbench.weekly.draft import compute_draft
from formbench.weekly.state import RegisterState

DOSES = [c.key for c in DOSE_COLUMNS]


def oracle_decisions(draft: dict[str, Any], pages: list[dict[str, Any]], photos: list[dict[str, Any]],
                     week_truth: dict[str, Any], intended_option: str) -> dict[str, Any]:  # fmt: skip
    """Resolve every open blocking flag the way a reviewer with the truth would."""
    decisions: dict[str, Any] = {"pages": {}, "resolved": {}}
    by_page = {p["page_id"]: (p, photo) for p, photo in zip(pages, photos)}
    for flag in draft["flags"]:
        if not flag["blocking"] or flag["resolved"]:
            continue
        code, pid, serial, column = flag["code"], flag["page"], flag["serial"], flag["column"]
        if code in ("facility_type_unclear", "attribute_option_invalid"):
            decisions["attribute_option"] = intended_option
            continue
        if pid not in by_page:
            decisions["resolved"][flag["id"]] = "oracle"
            continue
        _, photo = by_page[pid]
        truth_page = photo["reading"]
        true_fix = (
            week_truth["decisions"].get("pages", {}).get(photo["page_id"], {}).get("rows", {})
        )
        page_dec = decisions["pages"].setdefault(pid, {"rows": {}})
        if code == "cohort_unreadable":
            year, month = parse_cohort(truth_page["cohort_month"], truth_page["cohort_year"])
            page_dec["cohort"] = f"{year}-{month:02d}"
            continue
        truth_row = next((r for r in truth_page["rows"] if r["serial"] == serial), None)
        if truth_row is None:  # an invented row: tell the system it is not real
            decisions["resolved"][flag["id"]] = "oracle: not a real row"
            for key in DOSES:
                page_dec["rows"].setdefault(serial, {}).setdefault("cells", {})[key] = None
            continue
        row_dec = page_dec["rows"].setdefault(serial, {})
        fix = true_fix.get(serial, {})
        if code in ("birth_date_missing", "birth_date_unreadable"):
            row_dec["birth_date"] = fix.get("birth_date") or truth_row["birth_date"]
        elif code == "sex_unreadable":
            row_dec["sex"] = truth_row["sex"]
        elif code == "wrong_cohort":
            if fix.get("accept_cohort"):
                row_dec["accept_cohort"] = True
            else:
                row_dec["birth_date"] = truth_row["birth_date"]
                row_dec["accept_cohort"] = True
        elif code in ("date_unreadable", "future_date", "before_birth", "changed_since_approved") and column:  # fmt: skip
            value = (fix.get("cells") or {}).get(column, truth_row.get(column))
            if column in ("birth_date", "sex"):
                row_dec[column] = value
            else:
                row_dec.setdefault("cells", {})[column] = value
            if code == "changed_since_approved":
                decisions["resolved"][flag["id"]] = "oracle"
        else:
            decisions["resolved"][flag["id"]] = "oracle"
    return decisions


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(base))
    if extra.get("attribute_option"):
        out["attribute_option"] = extra["attribute_option"]
    out.setdefault("resolved", {}).update(extra.get("resolved", {}))
    for pid, page in extra.get("pages", {}).items():
        target = out.setdefault("pages", {}).setdefault(pid, {"rows": {}})
        if "cohort" in page:
            target["cohort"] = page["cohort"]
        for serial, row in page.get("rows", {}).items():
            merged = target.setdefault("rows", {}).setdefault(serial, {})
            cells = {**merged.get("cells", {}), **row.get("cells", {})}
            merged.update(row)
            if cells:
                merged["cells"] = cells
    return out


def review_and_approve(pages, photos, cover, spec, labels, week_truth, *, go_live, intended_option, state_root):  # fmt: skip
    """Draft -> oracle review until no blocking flag is open -> approve. Returns the
    first draft (what the reviewer saw) and the approved one."""
    week = week_truth["week"]
    today = dt.date.fromisoformat(week_truth["today"])

    def draft_with(decisions):
        return compute_draft(pages, cover, spec, labels, week=week, go_live=go_live, today=today,
                             state_root=state_root, decisions=decisions)  # fmt: skip

    decisions: dict[str, Any] = {}
    first = current = draft_with(decisions)
    for _ in range(4):
        if not current["unresolved_blocking"]:
            break
        decisions = _merge(decisions, oracle_decisions(current, pages, photos, week_truth, intended_option))  # fmt: skip
        current = draft_with(decisions)
    if current["unresolved_blocking"]:  # should not happen; resolve the rest blindly
        decisions["resolved"].update(
            {fid: "oracle: forced" for fid in current["unresolved_blocking"]}
        )
        current = draft_with(decisions)
    RegisterState(current["book"], root=state_root).approve(week, current["pages"], "oracle", "bench", week_truth["today"])  # fmt: skip
    return first, current


def flag_catch(drafts: list[dict[str, Any]], edges: list[dict[str, Any]], cover_flag_expected: bool) -> dict[str, int]:  # fmt: skip
    """Injected problems that should raise a blocking flag, and how many of them did."""
    expected = {(e["code"], e["serial"]) for e in edges if e["code"] in ("birth_date_missing", "wrong_cohort", "future_date")}  # fmt: skip
    raised = {(f["code"], f["serial"]) for d in drafts for f in d["flags"] if f["blocking"]}
    caught = {e for e in expected if e in raised}
    if cover_flag_expected:
        expected.add(("facility_type_unclear", None))
        if any(f["code"] == "facility_type_unclear" for d in drafts for f in d["flags"]):
            caught.add(("facility_type_unclear", None))
    return {"edges": len(expected), "edges_caught": len(caught),
            "blocking_flags": sum(len(d["unresolved_blocking"]) for d in drafts)}  # fmt: skip


ROW_FLAGS = {"birth_date_missing", "birth_date_unreadable", "sex_unreadable", "wrong_cohort",
             "duplicate_serial", "row_missing"}  # fmt: skip
CHECKED = ("sex", "birth_date", *DOSES)


def review_counts(flags: list[dict[str, Any]], pages: list[dict[str, Any]], photos: list[dict[str, Any]],
                  edges: list[dict[str, Any]]) -> dict[str, int]:  # fmt: skip
    """Did flags point the reviewer at the model's misreads, and how many flags were raised on
    readings that were actually right (false alarms)?

    A row-level flag (missing birth date, wrong cohort, ...) puts the whole row in front of the
    reviewer, so it covers every cell of that row; a cell flag covers its cell.
    """
    edge_rows = {(e["code"], e["serial"]) for e in edges}
    misread: set[tuple[str, str, str]] = set()
    truth_rows: dict[str, dict[str, Any]] = {}
    for page, photo in zip(pages, photos):
        truth = {r["serial"]: r for r in photo["reading"]["rows"]}
        truth_rows[page["page_id"]] = truth
        for row in page["rows"]:
            serial = str(row.get("serial") or "")
            t = truth.get(serial)
            for key in CHECKED:
                if t is None:  # invented row: anything it says is a misread
                    if not _blank(row.get(key)):
                        misread.add((page["page_id"], serial, key))
                elif not same_value(t.get(key), row.get(key), key):
                    misread.add((page["page_id"], serial, key))
    covered: set[tuple[str, str, str]] = set()
    false_alarms = 0
    for f in flags:
        if not f["blocking"] or not f["page"]:
            continue
        pid, serial = f["page"], f["serial"]
        if f["code"] == "cohort_unreadable":
            cells = {m for m in misread if m[0] == pid}
            useful = True
        elif f["code"] in ROW_FLAGS or not f["column"]:
            cells = {m for m in misread if m[:2] == (pid, serial)}
            useful = (
                bool(cells)
                or (f["code"], serial) in edge_rows
                or serial not in truth_rows.get(pid, {})
            )
        elif f["code"] == "changed_since_approved":
            # Two readings of one cell disagree, so one of them is wrong: if today's is right,
            # the flag exposes a wrong value approved earlier. Never a false alarm.
            cells = {(pid, serial, f["column"])} & misread
            useful = True
        else:
            cells = {(pid, serial, f["column"])} & misread
            useful = bool(cells) or (f["code"], serial) in edge_rows
        covered |= cells
        false_alarms += not useful
    return {
        "misreads": len(misread),
        "misreads_flagged": len(covered),
        "false_alarms": false_alarms,
    }
