from __future__ import annotations

import datetime as dt
import re

# Registers ask for DD-MM-YY; people also write DD/MM/YY, D.M.YYYY or wrap the year onto a
# second line, which reaches us as whitespace.
_DATE = re.compile(r"^(\d{1,2})\s*[-/.\s]\s*(\d{1,2})\s*[-/.\s]\s*(\d{2}|\d{4})$")
# Age notation from the register's guide: 7 (years), 3/12 (months), 6/52 (weeks), 5/7 (days).
_AGE = re.compile(r"^(\d{1,2})\s*(?:/\s*(12|52|7))?$")
AGE_UNITS = {None: "years", "12": "months", "52": "weeks", "7": "days"}


class Unreadable(ValueError):
    """Text was written but is not a valid value; a human must look at it."""


def _blank(text: str | None) -> bool:
    return text is None or not text.strip()


def parse_date(text: str | None) -> dt.date | None:
    """A handwritten date, None when blank. Two-digit years are 20YY."""
    if _blank(text):
        return None
    match = _DATE.match(text.strip())
    if not match:
        raise Unreadable(f"not a date: {text!r}")
    day, month, year = (int(g) for g in match.groups())
    if year < 100:
        year += 2000
    try:
        return dt.date(year, month, day)
    except ValueError:
        raise Unreadable(f"impossible date: {text!r}") from None


def parse_age(text: str | None) -> tuple[int, str] | None:
    """Age as written in the register, e.g. '3/12' -> (3, 'months'); None when blank."""
    if _blank(text):
        return None
    match = _AGE.match(text.strip())
    if not match:
        raise Unreadable(f"not an age: {text!r}")
    return int(match.group(1)), AGE_UNITS[match.group(2)]


def completed_months(birth: dt.date, on: dt.date) -> int:
    """Whole months of age on a date: 0 until the first monthly birthday, and so on."""
    months = (on.year - birth.year) * 12 + on.month - birth.month
    return months - 1 if on.day < birth.day else months
