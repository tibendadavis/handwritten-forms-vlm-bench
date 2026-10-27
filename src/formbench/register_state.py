"""What a reviewer has already approved, kept locally per register book.

Used to catch a cell that reads differently from its approved reading, and to turn doses
that appear late for an approved week into a correction with the new full day totals.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from formbench.handwriting import Unreadable, parse_date
from formbench.register_aggregate import (
    Aggregation,
    Flag,
    aggregate,
    iso_week,
    parse_cohort,
    select_days,
)
from formbench.register_spec import DOSE_COLUMNS, RegisterSpec
from formbench.schema_gen import slugify

STATE_DIR = Path("data/private/register/state")
COMPARED = ("sex", "birth_date", *(c.key for c in DOSE_COLUMNS))


def book_id(mfl_code: str | None, attribute_option: str) -> str:
    return slugify(f"{mfl_code or 'no-mfl'} {attribute_option}")


def page_identity(page: dict[str, Any], fallback: str) -> str:
    """Book-local page id: cohort month + first serial (continuation pages start higher)."""
    try:
        year, month = parse_cohort(page.get("cohort_month"), page.get("cohort_year"))
    except Unreadable:
        return f"unknown-{fallback}"
    serials = [int(s) for r in page["rows"] if str(s := r.get("serial") or "").isdigit()]
    return f"{year}-{month:02d}-s{min(serials) if serials else 1}"


class RegisterState:
    def __init__(self, book: str, root: Path = STATE_DIR) -> None:
        self.dir = root / book
        self.pages_dir = self.dir / "pages"

    def approved_pages(self) -> dict[str, dict[str, Any]]:
        if not self.pages_dir.exists():
            return {}
        return {p.stem: json.loads(p.read_text()) for p in sorted(self.pages_dir.glob("*.json"))}

    def approved_weeks(self) -> dict[str, Any]:
        path = self.dir / "weeks.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def approve(
        self, week: str, pages: list[dict[str, Any]], reviewer: str, run: str, when: str
    ) -> None:
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        for page in pages:
            stored = {k: v for k, v in page.items() if k != "page_id"}
            path = self.pages_dir / f"{page['page_id']}.json"
            path.write_text(json.dumps(stored, indent=2, ensure_ascii=False) + "\n")
        weeks = self.approved_weeks()
        weeks[week] = {"reviewer": reviewer, "run": run, "approved_at": when}
        (self.dir / "weeks.json").write_text(json.dumps(weeks, indent=2, sort_keys=True) + "\n")


def _same(a: Any, b: Any, key: str) -> bool:
    a, b = (str(v).strip() if v not in (None, "") else None for v in (a, b))
    if key == "sex":
        return (a or "")[:1].upper() == (b or "")[:1].upper()
    try:
        return parse_date(a) == parse_date(b)
    except Unreadable:
        return a == b


def inherit_decisions(page: dict[str, Any], approved: dict[str, Any]) -> tuple[dict[str, Any], list[Flag]]:  # fmt: skip
    """Carry a reviewer's earlier fixes (e.g. a birth date from the child's card) into a
    re-read of the same row when the cell still reads blank or the same as before the fix."""
    page = {**page, "rows": [dict(r) for r in page["rows"]]}
    old_rows = {str(r.get("serial")): r for r in approved["rows"]}
    notes = []
    for row in page["rows"]:
        serial = str(row.get("serial"))
        old = old_rows.get(serial)
        if not old:
            continue
        for key in old.get("_decided", []):
            if row.get(key) in (None, "") or not _parses(row.get(key), key):
                row[key] = old.get(key)
                row["_decided"] = sorted({*row.get("_decided", []), key})
                notes.append(Flag("kept_decision", f"{key} kept from an earlier approval: {old.get(key)!r}", False, page["page_id"], serial, key))  # fmt: skip
        if old.get("_accept_cohort"):
            row["_accept_cohort"] = True
    return page, notes


def _parses(value: Any, key: str) -> bool:
    if key == "sex":
        return str(value).strip()[:1].upper() in ("F", "M")
    try:
        parse_date(value)
        return True
    except Unreadable:
        return False


def discrepancies(page: dict[str, Any], approved: dict[str, Any]) -> list[Flag]:
    """Cells that read differently from the approved reading of the same page.

    New writing in a cell that was empty is expected (a later dose); anything else that
    changed is either a misread now or then, so a human must look.
    """
    flags = []
    pid = page["page_id"]
    now = {str(r.get("serial")): r for r in page["rows"]}
    for old in approved["rows"]:
        serial = str(old.get("serial"))
        if serial not in now:
            flags.append(Flag("row_missing", f"Row {serial} was approved before but is not read now", True, pid, serial))  # fmt: skip
            continue
        for key in COMPARED:
            before, after = old.get(key), now[serial].get(key)
            if before not in (None, "") and not _same(before, after, key):
                flags.append(Flag("changed_since_approved", f"{key}: approved {before!r}, now {after!r}", True, pid, serial, key))  # fmt: skip
    return flags


def corrections(
    pages: list[dict[str, Any]],
    state: RegisterState,
    spec: RegisterSpec,
    *,
    go_live: dt.date,
    today: dt.date,
    resolved: frozenset[str] = frozenset(),
) -> tuple[dict[tuple, int], Aggregation]:
    """Full new totals for days in approved weeks that gained doses since approval.

    Totals are recomputed from every approved page of the book with this run's readings
    replacing their earlier versions, because a DHIS2 import overwrites the stored value.
    """
    approved_weeks = set(state.approved_weeks())
    stored = state.approved_pages()
    if not approved_weeks or not stored:
        return {}, Aggregation()
    before = aggregate(
        [{"page_id": k, **v} for k, v in stored.items()], spec, go_live=go_live, today=today
    )
    merged = {k: {"page_id": k, **v} for k, v in stored.items()}
    merged.update({p["page_id"]: p for p in pages})
    after = aggregate(list(merged.values()), spec, go_live=go_live, today=today, resolved=resolved)
    known = {(d.page, d.serial, d.column, d.day) for d in before.doses}
    days = {
        d.day
        for d in after.doses
        if (d.page, d.serial, d.column, d.day) not in known and iso_week(d.day) in approved_weeks
    }
    return select_days(after, days), after
