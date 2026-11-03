"""A tally sheet described once, read one section photo at a time, scored field by field.

The form is described in a JSON file (form.json, kept out of git with the photos):

    {"id": "weekly-tally", "name": "Weekly tally sheet",
     "sections": {"A": {"title": "Session details",
                        "fields": [{"key": "Week starting", "type": "date"},
                                   {"key": "Region", "type": "choice", "options": ["North", "South"]}]},
                  "B": {"title": "Screening", "fields": [{"key": "Screened_Male", "type": "count"}]}}}

Field types: count (a number in digit boxes; empty means none), digits (a code, leading zeros
kept), text, date, choice (the option whose circle is filled), mark (signed or stamped, or not).

A truth file holds a whole filled sheet as field key -> value, whatever photos it was taken in;
a section photo is scored on its own section's fields only. A field is one cell in the scores:
right when the reading equals the truth after normalizing (numbers as numbers, dates as dates,
text ignoring case and spacing); empty on both sides is not counted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from formbench.register.handwriting import Unreadable, parse_date

TYPES = {"count", "digits", "text", "date", "choice", "mark"}
EMPTY = {"", "null", "none"}
HINTS = {
    "count": "number written in the boxes",
    "digits": "digits as written",
    "text": "text as written",
    "date": "date as written",
    "mark": '"signed" if signed or stamped, otherwise null',
}

# One prompt for every section, versioned in Langfuse; {{section}}, {{title}} and {{fields}}
# are filled in for the section in the photo.
DEFAULT_PROMPT = (
    'This is a photo of one section of a paper tally sheet: section {{section}}, "{{title}}". '
    "The section is framed by black squares at its corners. Numbers are handwritten one digit "
    "per box, right-aligned; empty boxes mean none. Return a JSON object with these keys:\n"
    "{{fields}}\n\n"
    "Rules:\n"
    "- Copy every number and word exactly as written; do not add up or correct anything.\n"
    "- Use null for a field with nothing written in it. Never guess.\n"
    "- For a list of circles, return the option whose circle is filled in.\n"
    "- Output only the JSON object."
)


def slug(key: str) -> str:
    """The JSON key the model returns for a field: "Screened_6-11 m_Male" -> screened_6_11_m_male."""
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


@dataclass(frozen=True)
class Field:
    key: str  # as on the form and in truth files
    type: str
    options: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return slug(self.key)

    def hint(self) -> str:
        if self.type == "choice":
            return f"the option whose circle is filled: {', '.join(self.options)}"
        return HINTS[self.type]


@dataclass(frozen=True)
class Section:
    letter: str
    title: str
    fields: tuple[Field, ...]


@dataclass(frozen=True)
class Form:
    id: str
    name: str
    sections: dict[str, Section]

    @property
    def prompt_name(self) -> str:
        return f"tally-{self.id}"

    def field(self, key: str) -> Field | None:
        return next(
            (f for s in self.sections.values() for f in s.fields if f.id == slug(key)), None
        )


def parse_form(data: dict[str, Any]) -> Form:
    """A form from its description; any mistake stops with what is wrong."""
    sections, seen = {}, set()
    for letter, raw in data["sections"].items():
        fields = []
        for f in raw["fields"]:
            field = Field(f["key"], f.get("type", "count"), tuple(f.get("options", ())))
            if field.type not in TYPES:
                raise ValueError(f"field {field.key!r}: type {field.type!r} is not one of {sorted(TYPES)}")  # fmt: skip
            if field.type == "choice" and not field.options:
                raise ValueError(f"field {field.key!r}: a choice needs its options")
            if field.id in seen:
                raise ValueError(f"field {field.key!r}: another field has the same name")
            seen.add(field.id)
            fields.append(field)
        sections[letter.upper()] = Section(letter.upper(), raw.get("title", letter), tuple(fields))
    return Form(data["id"], data.get("name", data["id"]), sections)


def find_form(folder: Path, explicit: Path | None = None) -> Path:
    """form.json given, in the folder, or in the folder above it."""
    for path in [explicit] if explicit else [folder / "form.json", folder.parent / "form.json"]:
        if path.exists():
            return path
    raise SystemExit(
        f"formbench tally: no form.json in {folder} or {folder.parent} (or give --form)"
    )


def load_form(path: Path) -> Form:
    try:
        return parse_form(json.loads(path.read_text()))
    except (ValueError, KeyError) as error:
        raise SystemExit(f"{path}: {error}") from None


def section_schema(section: Section) -> dict[str, Any]:
    props = {f.id: {"type": ["string", "null"], "description": f"{f.key}: {f.hint()}"} for f in section.fields}  # fmt: skip
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}  # fmt: skip


def prompt_variables(section: Section) -> dict[str, str]:
    fields = "\n".join(f"- {f.id}: {f.key.replace('_', ', ')} ({f.hint()})" for f in section.fields)
    return {"section": section.letter, "title": section.title, "fields": fields}


def compile_prompt(text: str, variables: dict[str, str]) -> str:
    """The prompt in code with its {{variables}} filled in (Langfuse prompts compile themselves)."""
    for name, value in variables.items():
        text = text.replace("{{" + name + "}}", value)
    return text


def load_sheet_truth(path: Path, form: Form) -> dict[str, Any]:
    """A whole filled sheet as field id -> value. An unknown field stops with its name."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path}: not valid JSON at line {error.lineno}, column {error.colno} "
                         f"({error.msg}). Numbers with leading zeros need quotes: \"0001\".") from None  # fmt: skip
    if isinstance(data, list) and len(data) == 1:
        data = data[0]
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: a truth file is one object of field -> value")
    out = {}
    for key, value in data.items():
        field = form.field(key)
        if field is None:
            raise SystemExit(f"{path}: unknown field {key!r}; fields are named as in form.json")
        out[field.id] = value
    return out


def blank(field: Field, value: Any) -> bool:
    """Nothing written: empty, null, or for a count zero (the form leaves none empty)."""
    text = "" if value is None else str(value).strip()
    if text.lower() in EMPTY:
        return True
    digits = re.sub(r"\s", "", text)
    return field.type == "count" and digits.isdigit() and int(digits) == 0


def same(field: Field, truth: Any, read: Any) -> bool:
    if blank(field, truth) or blank(field, read):
        return blank(field, truth) and blank(field, read)
    t, r = str(truth).strip(), str(read).strip()
    if field.type == "mark":
        return True
    if field.type == "count" and re.sub(r"\s", "", t).isdigit() and re.sub(r"\s", "", r).isdigit():
        return int(re.sub(r"\s", "", t)) == int(re.sub(r"\s", "", r))
    if field.type == "digits":
        return re.sub(r"\D", "", t) == re.sub(r"\D", "", r)
    if field.type == "date":
        try:
            return parse_date(t) == parse_date(r)
        except Unreadable:
            pass
    if field.type == "choice":
        return re.sub(r"[^a-z0-9]", "", t.casefold()) == re.sub(r"[^a-z0-9]", "", r.casefold())
    return " ".join(t.split()).casefold() == " ".join(r.split()).casefold()


def score_section(read: dict[str, Any], truth: dict[str, Any], section: Section, json_valid: bool) -> dict[str, float]:  # fmt: skip
    """Counts for one section photo, the same counts the register scores are built from."""
    out = {"pages": 1, "json_valid": float(json_valid), "field_cells": 0, "field_correct": 0,
           "field_tp": 0, "field_fp": 0, "field_fn": 0}  # fmt: skip
    for field in section.fields:
        t, r = truth.get(field.id), read.get(field.id)
        if blank(field, t) and blank(field, r):
            continue
        correct = same(field, t, r)
        out["field_cells"] += 1
        out["field_correct"] += correct
        out["field_tp"] += correct
        out["field_fp"] += not blank(field, r) and not correct
        out["field_fn"] += not blank(field, t) and not correct
    return out


def read_section(photo: Path, form: Form, letter: str, model: str, *, think: bool = False,
                 max_side: int, prompt: Any = None) -> dict[str, Any]:  # fmt: skip
    """Read one section photo with a Langfuse prompt (None: the prompt in code)."""
    from formbench.prompt_store import prompt_version
    from formbench.vlm import read_json

    section = form.sections[letter]
    variables = prompt_variables(section)
    text = (
        compile_prompt(DEFAULT_PROMPT, variables) if prompt is None else prompt.compile(**variables)
    )
    result = read_json(
        photo, section_schema(section), text, model, think=think, max_side=max_side,
        name=form.prompt_name, metadata={"prompt_version": prompt_version(prompt), "section": letter},
        langfuse_prompt=prompt,
    )  # fmt: skip
    reading = {k: None if v is None or str(v).strip().lower() in EMPTY else str(v).strip()
               for k, v in result["values"].items()}  # fmt: skip
    return {"reading": reading, "json_valid": result["json_valid"], "errors": result["errors"],
            "raw": result["raw"]}  # fmt: skip
