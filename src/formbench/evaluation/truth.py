"""Truth files for real photos: what is really written on each page.

Put the truth next to its photo: page.jpg + page.json (or page.truth.json). Either shape works:

    {"cohort_month": "7", "cohort_year": "2026", "rows": [{...}, {...}]}
    [{...}, {...}]                       # rows only
    {"1": {...}, "2": {...}}             # rows keyed by serial number

Each row is one child, as column -> value. Name a column by its key (bcg), its letter (I) or
both (I_bcg, "bcg (I)", "I BCG"). Column C holds two values, so name them birth_date and age.
Blank cells can be left out, null or "". Columns the model does not read (B first visit,
E names, F guardian and phone, H village, Z to AC) are ignored. The cohort month and year are
scored only if the truth has them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from formbench.register.prompts import ROW_FIELDS
from formbench.register.spec import DOSE_COLUMNS

NOT_READ = {"B", "E", "F", "H", "Z", "AA", "AB", "AC"}
PERSONAL = ("name", "guardian", "mother", "father", "phone", "village", "address", "contact")
HEADER = {"cohortmonth": "cohort_month", "month": "cohort_month",
          "cohortyear": "cohort_year", "year": "cohort_year"}  # fmt: skip


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


# Column key -> its letter (C is shared by birth date and age) and the names it may go by.
COLUMNS: dict[str, tuple[str, set[str]]] = {
    "serial": ("A", {"serial", "serialnumber", "serialno", "sno"}),
    "card_number": ("D", {"cardnumber", "cardno", "card", "childcardnumber", "childcardno"}),
    "sex": ("G", {"sex", "gender"}),
    "birth_date": ("C", {"birthdate", "dateofbirth", "dob"}),
    "age": ("C", {"age"}),
    **{c.key: (c.letter, {_norm(c.key), _norm(c.label)}) for c in DOSE_COLUMNS},
}
BY_NAME = {name: key for key, (_, names) in COLUMNS.items() for name in names}
BY_LETTER = {letter: key for key, (letter, _) in COLUMNS.items() if letter != "C"}
LETTERS = {letter for letter, _ in COLUMNS.values()} | NOT_READ


class TruthError(ValueError):
    pass


def column(raw: str) -> str | None:
    """The column key a truth file's key means; None for a column that is never scored."""
    name = _norm(raw)
    if name in BY_NAME:
        return BY_NAME[name]
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", raw) if p]
    letter = next((p.upper() for p in (parts[:1] + parts[-1:]) if p.upper() in LETTERS), None)
    if letter and len(parts) > 1:
        rest = [p for p in parts if p.upper() != letter]
        key = BY_NAME.get(_norm("".join(rest)))
        if key is not None:
            if COLUMNS[key][0] != letter:
                raise TruthError(f"{raw!r}: column {letter} is not {key} (that is column {COLUMNS[key][0]})")  # fmt: skip
            return key
        if (
            letter in BY_LETTER
        ):  # a name written another way ("Rotavirus_dose_1_W"): the letter decides
            return BY_LETTER[letter]
    if name.upper() == "C":
        raise TruthError(f"{raw!r}: column C holds two values; call them birth_date and age")
    if name.upper() in BY_LETTER:
        return BY_LETTER[name.upper()]
    if name.upper() in NOT_READ or letter in NOT_READ or any(p in name for p in PERSONAL):
        return None
    known = ", ".join(f"{key} ({letter})" for key, (letter, _) in COLUMNS.items())
    raise TruthError(f"unknown column {raw!r}. Columns: {known}")


def _value(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def normalize(data: Any) -> dict[str, Any]:
    """A truth file's content as a page reading: header keys, then rows with every key."""
    header, rows = {}, data
    if isinstance(data, dict) and "rows" not in data and data and all(isinstance(v, dict) for v in data.values()):  # fmt: skip
        rows = [{"serial": serial, **row} for serial, row in data.items()]  # keyed by serial
    elif isinstance(data, dict):
        rows = data.get("rows")
        for raw, value in data.items():
            if _norm(raw) in HEADER:
                header[HEADER[_norm(raw)]] = _value(value)
    if not isinstance(rows, list):
        raise TruthError('a truth file is a list of rows, or an object with "rows": [...]')
    out = []
    for n, row in enumerate(rows, 1):
        clean: dict[str, Any] = dict.fromkeys(ROW_FIELDS)
        for raw, value in row.items():
            try:
                key = column(raw)
            except TruthError as error:
                raise TruthError(f"row {n}: {error}") from None
            if key is not None:
                clean[key] = _value(value)
        out.append(clean)
    return {**header, "rows": out}


def truth_file(photo: Path) -> Path | None:
    """page.json or page.truth.json next to page.jpg."""
    for name in (f"{photo.stem}.truth.json", f"{photo.stem}.json"):
        if (photo.parent / name).exists():
            return photo.parent / name
    return None


def load_truth(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path}: not valid JSON at line {error.lineno}, column {error.colno} "
                         f"({error.msg}). Numbers with leading zeros need quotes: \"0001\".") from None  # fmt: skip
    try:
        return normalize(data)
    except TruthError as error:
        raise SystemExit(f"{path}: {error}") from None
