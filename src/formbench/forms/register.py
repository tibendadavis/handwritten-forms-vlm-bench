"""The register kind: a printed table, one row per patient, read one page photo per call.

Everything about the columns comes from form.json. The schema has no name, guardian, phone or
village fields, so the model cannot return them; personal columns in truth files are dropped
when the truth is loaded, so they never reach Langfuse or results files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from formbench import model
from formbench.forms.base import (
    FormError,
    ReadOptions,
    add_cell,
    empty_counts,
    find_truth,
    norm,
    outcome,
    photos_in,
    read_json_file,
    truth_files,
)
from formbench.matching import anls, match_lists
from formbench.prompts import REGISTER_PAGE_PROMPT, compile_prompt, prompt_version
from formbench.values import blank, clean_value, cohort, same_value

COLUMN_TYPES = ("text", "date", "sex")
HEADER_KEYS = ("cohort_month", "cohort_year")
SHARED_LETTER = "C"  # column C holds two values: birth date (top line) and age (bottom line)
TEXT = ["string", "null"]


@dataclass(frozen=True)
class Column:
    key: str
    letter: str
    label: str
    type: str
    description: str
    names: frozenset[str]  # every normalized name the column goes by in truth files


@dataclass(frozen=True)
class HeaderField:
    key: str
    description: str
    names: frozenset[str]


def _column(raw: dict[str, Any], n: int) -> Column:
    where = f"columns[{n}]"
    for attribute in ("key", "letter", "type", "description"):
        if not raw.get(attribute):
            raise FormError(f"{where}: {attribute!r} is missing")
    key, column_type = raw["key"], raw["type"]
    if column_type not in COLUMN_TYPES:
        raise FormError(f"{where} ({key}): type {column_type!r} is not one of {COLUMN_TYPES}")
    label = raw.get("label") or key
    names = frozenset({norm(key), norm(label), *(norm(name) for name in raw.get("names", []))})
    return Column(key, raw["letter"].upper(), label, column_type, raw["description"], names)


def _header(raw: dict[str, Any], n: int) -> HeaderField:
    key = raw.get("key")
    if key not in HEADER_KEYS:
        raise FormError(f"header[{n}]: key {key!r} is not one of {HEADER_KEYS}")
    # Truth files may say "month" for cohort_month: the last word of the key.
    names = frozenset({norm(key), norm(key.split("_")[-1])})
    return HeaderField(key, raw.get("description", key), names)


def _resolve(raw: str, form: RegisterForm) -> str | None:
    """The column key a truth file's key means, or None for a column that is never scored
    (this is also how personal columns are dropped). See the truth-file rules in the README."""
    name = norm(raw)
    if name in form.by_name:  # 1. a known name
        return form.by_name[name]
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", raw) if p]
    ends = parts[:1] + parts[-1:]
    letter = next((p.upper() for p in ends if p.upper() in form.letters), None)
    if letter and len(parts) > 1:  # 2. a name plus a letter
        rest = norm("".join(p for p in parts if p.upper() != letter))
        key = form.by_name.get(rest)
        if key is not None:
            column = form.column(key)
            if column.letter != letter:
                raise FormError(
                    f"{raw!r}: column {letter} is not {key} (that is column {column.letter})"
                )
            return key
        if letter in form.by_letter:  # a name written another way: the letter decides
            return form.by_letter[letter]
    if name.upper() == SHARED_LETTER:  # 3. plain C
        raise FormError(f"{raw!r}: column C holds two values; call them birth_date and age")
    if name.upper() in form.by_letter:  # 4. a plain letter
        return form.by_letter[name.upper()]
    ignored = form.not_read_letters
    if name.upper() in ignored or letter in ignored or any(w in name for w in form.personal):
        return None  # 5. never scored
    known = ", ".join(f"{c.key} ({c.letter})" for c in form.columns)
    raise FormError(f"unknown column {raw!r}. Columns: {known}")  # 6.


@dataclass(frozen=True)
class RegisterForm:
    id: str
    name: str
    prompt_name: str
    header: tuple[HeaderField, ...]
    columns: tuple[Column, ...]
    not_read_letters: frozenset[str]
    personal: tuple[str, ...]
    kind: str = "register"

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(c.key for c in self.columns)

    @property
    def by_name(self) -> dict[str, str]:
        return {name: c.key for c in self.columns for name in c.names}

    @property
    def by_letter(self) -> dict[str, str]:
        return {c.letter: c.key for c in self.columns if c.letter != SHARED_LETTER}

    @property
    def letters(self) -> set[str]:
        return {c.letter for c in self.columns} | set(self.not_read_letters)

    def column(self, key: str) -> Column:
        return next(c for c in self.columns if c.key == key)

    # truth files

    def normalize_truth(self, data: Any) -> dict[str, Any]:
        """A truth file's content as a page reading: header keys found, then rows that each
        have every column key. Personal and unread columns are dropped here."""
        header: dict[str, Any] = {}
        rows = data
        if (
            isinstance(data, dict)
            and "rows" not in data
            and data
            and all(isinstance(v, dict) for v in data.values())
        ):
            rows = [{"serial": serial, **row} for serial, row in data.items()]  # keyed by serial
        elif isinstance(data, dict):
            rows = data.get("rows")
            for raw, value in data.items():
                field = next((h for h in self.header if norm(raw) in h.names), None)
                if field is not None:
                    header[field.key] = _truth_value(value)
        if not isinstance(rows, list):
            raise FormError('a truth file is a list of rows, or an object with "rows": [...]')
        out = []
        for n, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise FormError(f"row {n}: a row is an object of column -> value")
            clean: dict[str, Any] = dict.fromkeys(self.keys)
            for raw, value in row.items():
                try:
                    key = _resolve(str(raw), self)
                except FormError as error:
                    raise FormError(f"row {n}: {error}") from None
                if key is not None:
                    clean[key] = _truth_value(value)
            out.append(clean)
        return {**header, "rows": out}

    def load_truth(self, path: Path) -> dict[str, Any]:
        data = read_json_file(path)
        try:
            return self.normalize_truth(data)
        except FormError as error:
            raise FormError(f"{path}: {error}") from None

    def check_truths(self, folder: Path) -> list[str]:
        errors = []
        for path in truth_files(folder):
            try:
                self.load_truth(path)
            except FormError as error:
                errors.append(str(error))
        return errors

    def items(self, folder: Path, pages: bool = False) -> tuple[list[dict], list[str]]:
        if pages:
            raise FormError("--pages is for cv-forms; a register is always one photo per page")
        items, skipped = [], []
        for photo in photos_in(folder):
            truth = find_truth(folder, photo.stem)
            if truth is None:
                skipped.append(f"{photo.name} (no truth file)")
                continue
            items.append(
                {
                    "id": f"{folder.name}/{photo.name}",
                    "input": {"photo": photo.name, "kind": "page"},
                    "expected_output": self.load_truth(truth),
                    "metadata": {"kind": "page"},
                }
            )
        return items, skipped

    # reading

    def schema(self) -> dict[str, Any]:
        """The reply shape for one page. Key order matters: it is the order the model sees."""
        row_props = {c.key: {"type": TEXT, "description": c.description} for c in self.columns}
        row = {
            "type": "object",
            "properties": row_props,
            "required": list(row_props),
            "additionalProperties": False,
        }
        props: dict[str, Any] = {
            h.key: {"type": TEXT, "description": h.description} for h in self.header
        }
        props["rows"] = {"type": "array", "items": row}
        return {
            "type": "object",
            "properties": props,
            "required": list(props),
            "additionalProperties": False,
        }

    def prompt_spec(self, pages: bool = False) -> tuple[str, str]:
        return self.prompt_name, REGISTER_PAGE_PROMPT

    def clean_rows(self, rows: Any) -> list[dict[str, Any]]:
        """Keep only dict rows and column keys, values as text or None; drop empty rows."""
        out = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            clean = {k: clean_value(row.get(k)) for k in self.keys}
            if any(clean.values()):
                out.append(clean)
        return out

    def read(
        self, folder: Path, item_input: dict, model_tag: str, prompt: Any, options: ReadOptions
    ) -> dict:
        result = model.read_json(
            folder / item_input["photo"],
            self.schema(),
            compile_prompt(prompt, {}),
            model_tag,
            think=options.think,
            max_side=options.max_side,
            trace_name=self.prompt_name,
            metadata={"prompt_version": prompt_version(prompt)},
            langfuse_prompt=prompt,
        )
        values = result["values"]
        reading = {h.key: values.get(h.key) for h in self.header}
        reading["rows"] = self.clean_rows(values.get("rows"))
        return {
            "reading": reading,
            "json_valid": result["json_valid"],
            "errors": result["errors"],
            "raw": result["raw"],
        }

    # scoring

    def pairing_text(self, key: str, value: Any) -> str:
        """A value as compared when pairing rows: dates as digits only, others with
        whitespace collapsed."""
        if blank(value):
            return ""
        if self.column(key).type == "date":
            return re.sub(r"\D", "", str(value))
        return " ".join(str(value).split())

    def row_similarity(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        """Mean ANLS over the columns either row fills; 1.0 when neither fills any."""
        texts = [
            (self.pairing_text(k, a.get(k)), self.pairing_text(k, b.get(k))) for k in self.keys
        ]
        texts = [(x, y) for x, y in texts if x or y]
        if not texts:
            return 1.0
        return sum(anls(x, y) for x, y in texts) / len(texts)

    def cell_pairs(self, reading: dict, truth: dict) -> list[tuple[str, Any, Any]]:
        """(key, truth value, read value) for every cell, rows paired by content so one
        invented or missing row does not shift the rest. Header cells only when the truth
        has them."""
        pairs = [
            (h.key, truth.get(h.key), reading.get(h.key)) for h in self.header if h.key in truth
        ]
        t_rows, r_rows = truth["rows"], reading.get("rows") or []
        matched = match_lists(t_rows, r_rows, self.row_similarity)
        seen_t, seen_r = {i for i, _, _ in matched}, {j for _, j, _ in matched}
        for i, j, _ in matched:
            pairs += [(k, t_rows[i].get(k), r_rows[j].get(k)) for k in self.keys]
        for i in (i for i in range(len(t_rows)) if i not in seen_t):
            pairs += [(k, t_rows[i].get(k), None) for k in self.keys]
        for j in (j for j in range(len(r_rows)) if j not in seen_r):
            pairs += [(k, None, r_rows[j].get(k)) for k in self.keys]
        return pairs

    def score(
        self, item_input: dict, reading: dict, expected: dict, json_valid: bool
    ) -> tuple[dict[str, float], list[dict]]:
        counts = empty_counts(json_valid)
        outcomes = []
        header_keys = {h.key for h in self.header}
        for key, t, r in self.cell_pairs(reading, expected):
            t_blank, r_blank = blank(t), blank(r)
            if t_blank and r_blank:
                continue
            if key in header_keys:  # month and year are right when they give the same cohort
                correct, group, field_type = cohort(expected) == cohort(reading), "header", "header"
            else:
                field_type = self.column(key).type
                correct, group = same_value(t, r, field_type), key
            add_cell(counts, correct, t_blank, r_blank)
            outcomes.append(
                outcome(
                    key,
                    group,
                    field_type,
                    t,
                    r,
                    correct=correct,
                    truth_blank=t_blank,
                    read_blank=r_blank,
                )
            )
        serials = {str(row.get("serial") or "").strip() for row in reading.get("rows") or []}
        truth_serials = [str(row.get("serial") or "").strip() for row in expected["rows"]]
        counts["rows"] = len(truth_serials)
        counts["rows_found"] = sum(s in serials for s in truth_serials if s)
        return counts, outcomes

    def is_empty_reading(self, item_input: dict, reading: dict, expected: dict) -> list[str]:
        read_any = reading.get("rows") or any(not blank(reading.get(h.key)) for h in self.header)
        truth_any = any(not blank(v) for row in expected["rows"] for v in row.values()) or any(
            not blank(expected.get(h.key)) for h in self.header
        )
        return ["page"] if truth_any and not read_any else []


def _truth_value(value: Any) -> str | None:
    """A truth value as stripped text; blank becomes None."""
    return None if blank(value) else str(value).strip()


def parse_register(data: dict[str, Any]) -> RegisterForm:
    """A register from its form.json; any mistake stops with the field and what is wrong."""
    for attribute in ("id", "columns"):
        if not data.get(attribute):
            raise FormError(f"{attribute!r} is missing")
    columns = tuple(_column(raw, n) for n, raw in enumerate(data["columns"]))
    header = tuple(_header(raw, n) for n, raw in enumerate(data.get("header", [])))
    keys = [c.key for c in columns] + [h.key for h in header]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise FormError(f"columns: key {duplicates[0]!r} is used more than once")
    if "serial" not in keys:
        raise FormError("columns: a column with key 'serial' is needed (rows are found by it)")
    for column in columns:
        shared = [c.key for c in columns if c.letter == column.letter]
        if column.letter != SHARED_LETTER and len(shared) > 1:
            raise FormError(f"columns ({column.key}): letter {column.letter} is also {shared}")
    return RegisterForm(
        id=data["id"],
        name=data.get("name", data["id"]),
        prompt_name=data.get("prompt", "register-page"),
        header=header,
        columns=columns,
        not_read_letters=frozenset(str(x).upper() for x in data.get("not_read_letters", [])),
        personal=tuple(norm(w) for w in data.get("personal_words", [])),
    )
