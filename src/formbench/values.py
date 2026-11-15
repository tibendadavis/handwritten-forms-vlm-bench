"""When two handwritten values count as the same: blanks, dates, numbers, codes and text.

Register cells and cv-form fields have their own rules, both kept here so every comparison the
scores depend on is in one place.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

# People write DD-MM-YY, DD/MM/YY, D.M.YYYY, or wrap the year onto a second line, which reaches
# us as whitespace.
DATE_PATTERN = re.compile(r"^(\d{1,2})\s*[-/.\s]\s*(\d{1,2})\s*[-/.\s]\s*(\d{2}|\d{4})$")
# English month names and 3-letter abbreviations -> month number, for the register header.
MONTHS = {
    name: number
    for number in range(1, 13)
    for name in (
        dt.date(2000, number, 1).strftime("%B").casefold(),
        dt.date(2000, number, 1).strftime("%b").casefold(),
    )
}
# Some models write an empty value as the text "null" instead of JSON null.
EMPTY_TEXT = {"", "null", "none"}


class Unreadable(ValueError):
    """Something was written, but it is not a valid value of its kind."""


def blank(value: Any) -> bool:
    """Nothing written: None or only whitespace (register cells)."""
    return value is None or not str(value).strip()


def clean_value(value: Any) -> str | None:
    """A model value as stripped text; "", "null" and "none" (any case) become None."""
    text = "" if value is None else str(value).strip()
    return None if text.lower() in EMPTY_TEXT else text


def parse_date(text: str | None) -> dt.date | None:
    """A handwritten day-month-year date; None when blank. Two-digit years are 20YY."""
    if blank(text):
        return None
    match = DATE_PATTERN.match(str(text).strip())
    if not match:
        raise Unreadable(f"not a date: {text!r}")
    day, month, year = (int(g) for g in match.groups())
    if year < 100:
        year += 2000
    try:
        return dt.date(year, month, day)
    except ValueError:
        raise Unreadable(f"impossible date: {text!r}") from None


def parse_cohort(month: str | None, year: str | None) -> tuple[int, int]:
    """The register page header -> (year, month). Accepts 'April', 'apr.', '4', '04' and
    '25', '2025'. Anything else raises Unreadable."""
    m = ("" if month is None else str(month)).strip().casefold().rstrip(".")
    y = ("" if year is None else str(year)).strip()
    number = int(m) if m.isdigit() else MONTHS.get(m)
    if not number or not 1 <= number <= 12 or not re.fullmatch(r"\d{2}|\d{4}", y):
        raise Unreadable(f"cohort month/year not readable: {month!r} {year!r}")
    return (int(y) + 2000 if len(y) == 2 else int(y)), number


def cohort(reading: dict[str, Any]) -> tuple[int, int] | None:
    """The (year, month) a reading's header gives, or None when it cannot be read."""
    try:
        return parse_cohort(reading.get("cohort_month"), reading.get("cohort_year"))
    except Unreadable:
        return None


def _words(value: Any) -> str:
    """Whitespace collapsed and case-folded, for comparing plain text."""
    return " ".join(str(value).split()).casefold()


def same_value(truth: Any, read: Any, column_type: str) -> bool:
    """Whether a register cell was read right: dates as calendar dates, sex by its first
    letter, anything else as text ignoring case and extra spaces."""
    if blank(truth) or blank(read):
        return blank(truth) and blank(read)
    if column_type == "sex":
        return str(truth).strip()[:1].upper() == str(read).strip()[:1].upper()
    if column_type == "date":
        try:
            return parse_date(str(truth)) == parse_date(str(read))
        except Unreadable:
            return False
    return _words(truth) == _words(read)


def _no_space(text: str) -> str:
    return re.sub(r"\s", "", text)


def field_blank(field_type: str, value: Any) -> bool:
    """Nothing written in a cv-form field: empty, null, or for a count zero (the form leaves no
    count box empty, so 0 and nothing mean the same)."""
    text = "" if value is None else str(value).strip()
    if text.lower() in EMPTY_TEXT:
        return True
    digits = _no_space(text)
    return field_type == "count" and digits.isdigit() and int(digits) == 0


def same(field_type: str, truth: Any, read: Any) -> bool:
    """Whether a cv-form field was read right, by the rules of its type (in this order)."""
    if field_blank(field_type, truth) or field_blank(field_type, read):
        return field_blank(field_type, truth) and field_blank(field_type, read)
    t, r = str(truth).strip(), str(read).strip()
    if field_type == "mark":
        return True  # both sides are signed or stamped
    if field_type == "count" and _no_space(t).isdigit() and _no_space(r).isdigit():
        return int(_no_space(t)) == int(_no_space(r))
    if field_type == "digits":
        return re.sub(r"\D", "", t) == re.sub(r"\D", "", r)
    if field_type == "date":
        try:
            return parse_date(t) == parse_date(r)
        except Unreadable:
            pass  # not a date on one side: compare as text below
    if field_type == "choice":
        return re.sub(r"[^a-z0-9]", "", t.casefold()) == re.sub(r"[^a-z0-9]", "", r.casefold())
    return _words(t) == _words(r)
