"""Register readings -> daily RI v2 counts. Pure: no I/O, no model, same input -> same output.

Every count is traceable to the register cells it came from, and anything a human must
decide becomes a flag instead of a guess.
"""

from __future__ import annotations

import copy
import datetime as dt
import difflib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from formbench.handwriting import Unreadable, completed_months, parse_date
from formbench.register_spec import DOSE_COLUMNS, RegisterSpec

MONTHS = {
    name: i
    for i in range(1, 13)
    for name in (
        dt.date(2000, i, 1).strftime("%B").casefold(),
        dt.date(2000, i, 1).strftime("%b").casefold(),
    )
}
# Attribute options a person must choose explicitly; never matched from handwriting.
MANUAL_ONLY = {"unclear", "not filled"}

Key = tuple  # (day, data element UID, category option combo UID)


@dataclass(frozen=True)
class Flag:
    code: str
    message: str
    blocking: bool
    page: str | None = None
    serial: str | None = None
    column: str | None = None

    @property
    def id(self) -> str:
        return ":".join(p for p in (self.code, self.page, self.serial, self.column) if p)


@dataclass(frozen=True)
class CountedDose:
    page: str
    serial: str
    column: str
    day: dt.date
    data_element: str
    coc: str


@dataclass
class Aggregation:
    doses: list[CountedDose] = field(default_factory=list)
    totals: dict[Key, int] = field(default_factory=dict)  # every day found, doses + children
    provenance: dict[Key, list[tuple[str, str, str]]] = field(default_factory=dict)
    flags: list[Flag] = field(default_factory=list)


def iso_week_days(week: str) -> list[dt.date]:
    """'2026W44' -> the seven days Monday..Sunday of that ISO week."""
    match = re.fullmatch(r"(\d{4})-?W(\d{1,2})", week)
    if not match:
        raise ValueError(f"Not an ISO week: {week!r} (expected e.g. 2026W44)")
    year, number = int(match.group(1)), int(match.group(2))
    monday = dt.date.fromisocalendar(year, number, 1)
    return [monday + dt.timedelta(days=i) for i in range(7)]


def iso_week(day: dt.date) -> str:
    year, number, _ = day.isocalendar()
    return f"{year}W{number}"


def parse_cohort(month: str | None, year: str | None) -> tuple[int, int]:
    """Page header Month/Year -> (year, month). Accepts 'April', 'Apr', '4', '04'; '25', '2025'."""
    m, y = (month or "").strip().casefold().rstrip("."), (year or "").strip()
    number = int(m) if m.isdigit() else MONTHS.get(m)
    if not number or not 1 <= number <= 12 or not re.fullmatch(r"\d{2}|\d{4}", y):
        raise Unreadable(f"cohort month/year not readable: {month!r} {year!r}")
    return (int(y) + 2000 if len(y) == 2 else int(y)), number


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def resolve_attribute_option(
    facility_type: str | None, spec: RegisterSpec, chosen: str | None = None
) -> tuple[str | None, Flag | None]:
    """Facility Type as handwritten -> attribute option combo UID, or a blocking flag."""
    if chosen:
        if chosen not in spec.attribute_options:
            return None, Flag("attribute_option_invalid", f"{chosen!r} is not an option", True)
        return spec.attribute_options[chosen], None
    options = {n: uid for n, uid in spec.attribute_options.items() if _norm(n) not in MANUAL_ONLY}
    written = _norm(facility_type or "")
    if written:
        exact = [n for n in options if _norm(n) == written]
        parts = [n for n in options if written in {_norm(p) for p in n.split("/")}]
        hits = exact or parts
        if len(hits) == 1:
            return options[hits[0]], None
    suggestion = difflib.get_close_matches(written, [_norm(n) for n in options], n=1)
    names = {_norm(n): n for n in options}
    hint = f"; closest option: {names[suggestion[0]]}" if suggestion else ""
    what = "blank" if not written else f"{facility_type!r} does not match an option"
    return None, Flag("facility_type_unclear", f"Facility Type is {what}{hint}", True)


def apply_decisions(pages: list[dict[str, Any]], decisions: dict[str, Any]) -> list[dict[str, Any]]:
    """Reviewer corrections applied on a copy: cohort, birth date, sex and cell values."""
    pages = copy.deepcopy(pages)
    for page in pages:
        page_decisions = (decisions.get("pages") or {}).get(page["page_id"]) or {}
        if "cohort" in page_decisions:
            year, month = str(page_decisions["cohort"]).split("-")
            page["cohort_year"], page["cohort_month"] = year, month
        for row in page["rows"]:
            row_decisions = (page_decisions.get("rows") or {}).get(str(row.get("serial"))) or {}
            for key in ("birth_date", "sex"):
                if key in row_decisions:
                    row[key] = row_decisions[key]
            row.update(row_decisions.get("cells") or {})
            row["_accept_cohort"] = bool(row_decisions.get("accept_cohort"))
    return pages


def aggregate(
    pages: list[dict[str, Any]],
    spec: RegisterSpec,
    *,
    go_live: dt.date,
    today: dt.date,
    resolved: frozenset[str] = frozenset(),
) -> Aggregation:
    """Count every dose by its own date. Rows or cells with a blocking problem are not counted.

    `pages` are readings with decisions already applied: {page_id, cohort_month, cohort_year,
    rows: [{serial, sex, birth_date, <dose keys>...}]}. `resolved` holds flag ids a reviewer
    accepted as they are (e.g. a duplicate serial that really is two children).
    """
    result = Aggregation()
    children: dict[tuple[dt.date, str], set[tuple[str, str]]] = defaultdict(set)

    def flag(code, message, blocking, page=None, serial=None, column=None):
        result.flags.append(Flag(code, message, blocking, page, serial, column))

    for page in pages:
        pid = page["page_id"]
        try:
            cohort = parse_cohort(page.get("cohort_month"), page.get("cohort_year"))
        except Unreadable as error:
            cohort = None
            flag("cohort_unreadable", str(error), True, pid)
        seen: set[str] = set()
        for index, row in enumerate(page["rows"], start=1):
            serial = str(row.get("serial") or "").strip() or f"row{index}"
            if not any(row.get(c.key) for c in DOSE_COLUMNS):
                continue
            if serial in seen:
                f = Flag("duplicate_serial", f"Serial {serial} appears twice", True, pid, serial)
                if f.id not in resolved:
                    result.flags.append(f)
                    continue
            seen.add(serial)
            if row.get("serial") in (None, ""):
                flag("serial_missing", "Row has no serial number", False, pid, serial)

            sex = (row.get("sex") or "").strip().upper()[:1]
            if sex not in ("F", "M"):
                flag("sex_unreadable", f"Sex is {row.get('sex')!r}", True, pid, serial)
            try:
                birth = parse_date(row.get("birth_date"))
                if birth is None:
                    flag(
                        "birth_date_missing",
                        "No birth date; age cannot be computed",
                        True,
                        pid,
                        serial,
                    )
            except Unreadable as error:
                birth = None
                flag("birth_date_unreadable", str(error), True, pid, serial)
            if cohort is None or birth is None or sex not in ("F", "M"):
                continue
            if (birth.year, birth.month) != cohort and not row.get("_accept_cohort"):
                flag(
                    "wrong_cohort",
                    f"Born {birth:%d-%m-%y}, but the page is the {cohort[1]:02d}/{cohort[0]} cohort",
                    True, pid, serial,
                )  # fmt: skip
                continue

            for column in DOSE_COLUMNS:
                text = row.get(column.key)
                try:
                    day = parse_date(text)
                except Unreadable as error:
                    flag("date_unreadable", str(error), True, pid, serial, column.key)
                    continue
                if day is None:
                    continue
                where = (pid, serial, column.key)
                if column.vaccine is None:
                    flag("unmapped_column", f"{column.label} {day:%d-%m-%y} is not counted (unmapped)", False, *where)  # fmt: skip
                    continue
                if day < go_live:
                    flag("before_go_live", f"{column.label} {day:%d-%m-%y} predates go-live", False, *where)  # fmt: skip
                    continue
                if day > today:
                    flag("future_date", f"{column.label} {day:%d-%m-%y} is in the future", True, *where)  # fmt: skip
                    continue
                if day < birth:
                    flag("before_birth", f"{column.label} {day:%d-%m-%y} is before birth", True, *where)  # fmt: skip
                    continue
                months = completed_months(birth, day)
                target = spec.vaccines[column.vaccine]
                coc = target.coc(column.dose, sex, months)
                if coc is None:
                    flag("no_age_band", f"{column.label} at {months} months has no DHIS2 age band; not counted", False, *where)  # fmt: skip
                    continue
                result.doses.append(
                    CountedDose(pid, serial, column.key, day, target.data_element, coc)
                )
                key = (day, target.data_element, coc)
                result.totals[key] = result.totals.get(key, 0) + 1
                result.provenance.setdefault(key, []).append(where)
                child_coc = spec.children.coc(None, sex, months)
                if child_coc:
                    children[(day, child_coc)].add((pid, serial))

    for (day, coc), rows in sorted(children.items()):
        key = (day, spec.children.data_element, coc)
        result.totals[key] = len(rows)
        result.provenance[key] = sorted((p, s, "*") for p, s in rows)
    result.totals = dict(sorted(result.totals.items()))
    return result


def select_days(agg: Aggregation, days: set[dt.date] | list[dt.date]) -> dict[Key, int]:
    wanted = set(days)
    return {k: v for k, v in agg.totals.items() if k[0] in wanted}


def classify_days(agg: Aggregation, week: str, approved_weeks: set[str]) -> dict[str, set[dt.date]]:
    """Which counted days belong to the reporting week, an approved earlier week (possible
    correction), an earlier week never reported, or a later week."""
    days = set(iso_week_days(week))
    out: dict[str, set[dt.date]] = {
        "week": set(),
        "approved": set(),
        "unreported": set(),
        "later": set(),
    }
    for dose in agg.doses:
        if dose.day in days:
            out["week"].add(dose.day)
        elif dose.day > max(days):
            out["later"].add(dose.day)
        elif iso_week(dose.day) in approved_weeks:
            out["approved"].add(dose.day)
        else:
            out["unreported"].add(dose.day)
    return out
