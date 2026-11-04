"""A tally sheet described once, read one section photo or one whole page at a time, scored
field by field.

The form is described in a JSON file (form.json, kept out of git with the photos):

    {"id": "weekly-tally", "name": "Weekly tally sheet",
     "sections": {"A": {"title": "Session details",
                        "fields": [{"key": "Week starting", "type": "date"},
                                   {"key": "Region", "type": "choice", "options": ["North", "South"]}]},
                  "B": {"title": "Screening", "fields": [{"key": "Screened_Male", "type": "count"}]}},
     "pages": ["A", "B"]}

"pages" lists the sections printed on each page of the paper form, page 1 first; without it the
form is one page holding every section.

Field types: count (a number in digit boxes; empty means none), digits (a code, leading zeros
kept), text, date, choice (the option whose circle is filled), mark (signed or stamped, or not).

A truth file holds a whole filled sheet as field key -> value, whatever photos it was taken in;
a section photo is scored on its own section's fields only, a page photo on its sections' fields. A field is one cell in the scores:
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
# A whole page in one call, versioned in Langfuse apart from the section prompt; {{fields}} lists
# the fields of every section on the page, section by section.
DEFAULT_PAGE_PROMPT = """\
This is a photo of a whole page of a paper tally sheet. The page has several sections. Each section is framed by black squares at its corners and headed by its letter and title, for example "A Session details". Return one JSON object with these keys, section by section:
{{fields}}

How to find each field:
- Find the section by its heading, then the field by its label on the left of its row. In a table, the column headings above give the age group and sex of each group of boxes; read the row from left to right.
- Each field has its own small group of boxes. Read only the boxes of that group. Never join digits from two neighbouring groups and never take a digit from the row above or below.

How the form is filled in:
- Numbers are handwritten one digit per box and right-aligned, so the left boxes are usually empty and the digits sit in the rightmost boxes. Read every box of a field from left to right and write only the digits you see; an empty box adds nothing. For example, boxes [ ][4][7] are 47 and [ ][ ][3] is 3. A single digit in the last box is still a number; do not skip it.
- Text is in block capitals, one letter per box. Read each box once: do not repeat a letter or add one that is not there. An empty box between letters is a space between words.
- Circles: the filled (dark) circle is the answer. If no circle in a list is filled, use null.
- Check each circle in a list; a filled circle is solid dark. The Day circles may all be empty, or have exactly one filled.

People write some characters differently. Read them carefully:
- A 7 may have a short bar across its middle. It is still 7, not 2 or 4.
- A 3 may have a flat top. It is still 3, not 2.
- A 6 may look like a C or G with a small loop at the bottom. It is still 6, not 5 or 0.
- A 1 may have a small flag at the top. It is still 1, not 7.
- An R may have a curly or bent leg. It is still R, not E or K.

Rules:
- Copy every number and word exactly as written; do not add up, correct or complete anything.
- If a character is hard to read, give your best reading. Use null only when all the boxes of a field are empty.
- A signature or stamp in the signature box means "signed".
- Output only the JSON object."""
# A page photo is enlarged to this long side: at phone-chat size (about 1200 px) a digit box is
# about 20 px, too few for the model to tell neighbouring boxes and rows apart.
PAGE_MIN_SIDE = 2560


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
    pages: tuple[tuple[str, ...], ...]  # the section letters printed on each page

    @property
    def prompt_name(self) -> str:
        return f"tally-{self.id}"

    @property
    def page_prompt_name(self) -> str:
        return f"tally-{self.id}-page"

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
    pages = tuple(tuple(str(letter).upper() for letter in page) for page in data.get("pages", [list(sections)]))  # fmt: skip
    for page in pages:
        for letter in page:
            if letter not in sections:
                raise ValueError(f"pages: section {letter!r} is not in the form")
    printed = [letter for page in pages for letter in page]
    if len(printed) != len(set(printed)):
        raise ValueError("pages: a section is on more than one page")
    return Form(data["id"], data.get("name", data["id"]), sections, pages)


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


def page_schema(sections: list[Section]) -> dict[str, Any]:
    """Every field of the sections on a page in one object (field names are unique in a form)."""
    props = {k: v for s in sections for k, v in section_schema(s)["properties"].items()}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}  # fmt: skip


def field_lines(section: Section) -> str:
    return "\n".join(f"- {f.id}: {f.key.replace('_', ', ')} ({f.hint()})" for f in section.fields)


def prompt_variables(section: Section) -> dict[str, str]:
    return {"section": section.letter, "title": section.title, "fields": field_lines(section)}


def page_prompt_variables(sections: list[Section]) -> dict[str, str]:
    return {"fields": "\n\n".join(f'Section {s.letter}, "{s.title}":\n{field_lines(s)}' for s in sections)}  # fmt: skip


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
    return {"reading": clean(result["values"]), "json_valid": result["json_valid"],
            "errors": result["errors"], "raw": result["raw"]}  # fmt: skip


def read_page(photo: Path, form: Form, page: int, model: str, *, think: bool = False,
              max_side: int, min_side: int = PAGE_MIN_SIDE, prompt: Any = None) -> dict[str, Any]:  # fmt: skip
    """Read every section of a whole-page photo in one call with the page prompt (None: the
    prompt in code); the reading is split per section, as if each had its own photo."""
    from formbench.prompt_store import prompt_version
    from formbench.vlm import read_json

    sections = [form.sections[letter] for letter in form.pages[page - 1]]
    variables = page_prompt_variables(sections)
    text = (
        compile_prompt(DEFAULT_PAGE_PROMPT, variables) if prompt is None else prompt.compile(**variables)
    )  # fmt: skip
    result = read_json(
        photo, page_schema(sections), text, model, think=think, max_side=max_side,
        min_side=min_side, name=form.page_prompt_name,
        metadata={"prompt_version": prompt_version(prompt), "page": page}, langfuse_prompt=prompt,
    )  # fmt: skip
    values = clean(result["values"])
    readings = {s.letter: {f.id: values.get(f.id) for f in s.fields} for s in sections}
    return {"readings": readings, "json_valid": result["json_valid"], "errors": result["errors"],
            "raw": result["raw"]}  # fmt: skip


def clean(values: dict[str, Any]) -> dict[str, str | None]:
    """Model values as text, with "null", "none" and blanks as empty fields."""
    return {k: None if v is None or str(v).strip().lower() in EMPTY else str(v).strip()
            for k, v in values.items()}  # fmt: skip
