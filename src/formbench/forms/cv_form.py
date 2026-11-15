"""The cv-form kind: a form printed in framed sections, one character per box, choices marked
by filling a circle. Photographed one section per photo (close-ups) or one whole page per photo.

A truth file holds a whole filled sheet as field key -> value, whatever photos it was taken in;
a section photo is scored on its own section's fields, a page photo on its page's sections.
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
    outcome,
    photos_in,
    read_json_file,
    truth_files,
)
from formbench.prompts import PAGE_PROMPT, SECTION_PROMPT, compile_prompt, prompt_version
from formbench.values import clean_value, field_blank, same

TYPES = ("count", "digits", "text", "date", "choice", "mark")
HINTS = {
    "count": "number written in the boxes",
    "digits": "digits as written",
    "text": "text as written",
    "date": "date as written",
    "mark": '"signed" if signed or stamped, otherwise null',
}
SECTION_PHOTO = re.compile(r"^(?P<sheet>.+)[_-](?P<section>[A-Za-z])$")
PAGE_PHOTO = re.compile(r"^(?P<sheet>.+?)(?:[_-]p(?P<page>\d+))?$")


def slug(key: str) -> str:
    """A field's id, the JSON key the model returns: "Screened_6-11 m_Male" ->
    "screened_6_11_m_male"."""
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


@dataclass(frozen=True)
class Field:
    key: str  # as printed on the form and used in truth files
    type: str
    options: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return slug(self.key)

    def hint(self) -> str:
        """What the model should return for this field, as shown in the prompt and schema."""
        if self.type == "choice":
            return f"the option whose circle is filled: {', '.join(self.options)}"
        return HINTS[self.type]


@dataclass(frozen=True)
class Section:
    letter: str
    title: str
    fields: tuple[Field, ...]


def field_lines(section: Section) -> str:
    """One prompt line per field: "- <id>: <key, with _ as ', '> (<hint>)"."""
    return "\n".join(f"- {f.id}: {f.key.replace('_', ', ')} ({f.hint()})" for f in section.fields)


def section_variables(section: Section) -> dict[str, str]:
    """The variables of the section prompt."""
    return {"section": section.letter, "title": section.title, "fields": field_lines(section)}


def page_variables(sections: list[Section]) -> dict[str, str]:
    """The variables of the page prompt: every section on the page, one after another."""
    blocks = [f'Section {s.letter}, "{s.title}":\n{field_lines(s)}' for s in sections]
    return {"fields": "\n\n".join(blocks)}


def schema_for(sections: list[Section]) -> dict[str, Any]:
    """One reply object with every field of the given sections, in form order."""
    props = {
        f.id: {"type": ["string", "null"], "description": f"{f.key}: {f.hint()}"}
        for s in sections
        for f in s.fields
    }
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class CvForm:
    id: str
    name: str
    sections: dict[str, Section]
    pages: tuple[str, ...]  # the section letters printed on each page, page 1 first
    section_prompt: str
    page_prompt: str
    kind: str = "cv-form"

    def field(self, key: str) -> Field | None:
        return next(
            (f for s in self.sections.values() for f in s.fields if f.id == slug(key)), None
        )

    def page_sections(self, page: int) -> list[Section]:
        return [self.sections[letter] for letter in self.pages[page - 1]]

    # truth files

    def load_sheet_truth(self, path: Path) -> dict[str, Any]:
        """A whole filled sheet as field id -> value. An unknown field stops with its name."""
        data = read_json_file(path)
        if isinstance(data, list) and len(data) == 1:
            data = data[0]
        if not isinstance(data, dict):
            raise FormError(f"{path}: a truth file is one object of field -> value")
        out = {}
        for key, value in data.items():
            field = self.field(str(key))
            if field is None:
                raise FormError(f"{path}: unknown field {key!r}; fields are named as in form.json")
            out[field.id] = value
        return out

    def check_truths(self, folder: Path) -> list[str]:
        errors = []
        for path in truth_files(folder):
            try:
                self.load_sheet_truth(path)
            except FormError as error:
                errors.append(str(error))
        return errors

    # photos

    def photo_part(self, stem: str) -> tuple[str, str, int | None] | None:
        """(sheet, section letters, page number or None for a section photo) from a photo's
        name; None when it is neither. The section pattern is checked first."""
        match = SECTION_PHOTO.match(stem)
        if match and match["section"].upper() in self.sections:
            return match["sheet"], match["section"].upper(), None
        match = PAGE_PHOTO.match(stem)
        page = int(match["page"] or 1) if match else 0
        if match and 1 <= page <= len(self.pages):
            return match["sheet"], self.pages[page - 1], page
        return None

    def naming_rule(self, pages: bool) -> str:
        if pages:
            return (
                f"name it <sheet>.jpg for page 1 or <sheet>_p<n>.jpg, n from 1 to {len(self.pages)}"
            )
        return f"name it <sheet>_<section>.jpg, sections {', '.join(self.sections)}"

    def items(self, folder: Path, pages: bool = False) -> tuple[list[dict], list[str]]:
        items, skipped, truths = [], [], {}
        for photo in photos_in(folder):
            found = self.photo_part(photo.stem)
            truth = find_truth(folder, found[0]) if found else None
            if found and (found[2] is not None) != pages:
                if truth is not None:
                    continue  # a photo of the other mode
                found = None  # fits neither mode for a sheet that has a truth
            if found is None:
                skipped.append(f"{photo.name} ({self.naming_rule(pages)})")
                continue
            sheet, letters, page = found
            if truth is None:
                skipped.append(f"{photo.name} (no {sheet}.json)")
                continue
            if sheet not in truths:
                truths[sheet] = self.load_sheet_truth(truth)
            expected = {
                f.id: truths[sheet].get(f.id)
                for letter in letters
                for f in self.sections[letter].fields
            }
            where = {"page": page, "sections": letters} if pages else {"section": letters}
            items.append(
                {
                    "id": f"{folder.name}/{photo.name}",
                    "input": {"photo": photo.name, **where},
                    "expected_output": expected,
                    "metadata": {"sheet": sheet, **where},
                }
            )
        return items, skipped

    # reading

    def prompt_spec(self, pages: bool = False) -> tuple[str, str]:
        return (self.page_prompt, PAGE_PROMPT) if pages else (self.section_prompt, SECTION_PROMPT)

    def letters(self, item_input: dict) -> str:
        """The section letters a photo holds."""
        return item_input.get("sections") or item_input["section"]

    def read(
        self, folder: Path, item_input: dict, model_tag: str, prompt: Any, options: ReadOptions
    ) -> dict:
        """One call per photo, a section or a whole page; the reading is always split per
        section: {letter: {field id: value}}."""
        version = prompt_version(prompt)
        if "page" in item_input:
            sections = self.page_sections(item_input["page"])
            variables = page_variables(sections)
            trace_name, metadata = (
                self.page_prompt,
                {"prompt_version": version, "page": item_input["page"]},
            )
        else:
            sections = [self.sections[item_input["section"]]]
            variables = section_variables(sections[0])
            trace_name = self.section_prompt
            metadata = {"prompt_version": version, "section": item_input["section"]}
        result = model.read_json(
            folder / item_input["photo"],
            schema_for(sections),
            compile_prompt(prompt, variables),
            model_tag,
            think=options.think,
            max_side=options.max_side,
            trace_name=trace_name,
            metadata=metadata,
            langfuse_prompt=prompt,
        )
        values = {k: clean_value(v) for k, v in result["values"].items()}
        reading = {s.letter: {f.id: values.get(f.id) for f in s.fields} for s in sections}
        return {
            "reading": reading,
            "json_valid": result["json_valid"],
            "errors": result["errors"],
            "raw": result["raw"],
        }

    # scoring

    def score(
        self, item_input: dict, reading: dict, expected: dict, json_valid: bool
    ) -> tuple[dict[str, float], list[dict]]:
        counts = empty_counts(json_valid)
        outcomes = []
        for letter in self.letters(item_input):
            read_section = reading.get(letter) or {}
            for f in self.sections[letter].fields:
                t, r = expected.get(f.id), read_section.get(f.id)
                t_blank, r_blank = field_blank(f.type, t), field_blank(f.type, r)
                if t_blank and r_blank:
                    continue
                correct = same(f.type, t, r)
                add_cell(counts, correct, t_blank, r_blank)
                outcomes.append(
                    outcome(
                        f.id,
                        letter,
                        f.type,
                        t,
                        r,
                        correct=correct,
                        truth_blank=t_blank,
                        read_blank=r_blank,
                    )
                )
        return counts, outcomes

    def is_empty_reading(self, item_input: dict, reading: dict, expected: dict) -> list[str]:
        empty = []
        for letter in self.letters(item_input):
            fields = self.sections[letter].fields
            read_section = reading.get(letter) or {}
            read_any = any(not field_blank(f.type, read_section.get(f.id)) for f in fields)
            truth_any = any(not field_blank(f.type, expected.get(f.id)) for f in fields)
            if truth_any and not read_any:
                empty.append(letter)
        return empty


def _field(raw: Any, where: str, seen: set[str]) -> Field:
    if not isinstance(raw, dict) or not raw.get("key"):
        raise FormError(f'{where}: a field is an object with a "key"')
    field = Field(str(raw["key"]), raw.get("type", "count"), tuple(raw.get("options", ())))
    where = f"{where} ({field.key})"
    if field.type not in TYPES:
        raise FormError(f"{where}: type {field.type!r} is not one of {TYPES}")
    if field.type == "choice" and not field.options:
        raise FormError(f'{where}: a choice needs its "options"')
    if not field.id:
        raise FormError(f"{where}: the key needs letters or digits")
    if field.id in seen:
        raise FormError(f"{where}: another field has the same id {field.id!r}")
    seen.add(field.id)
    return field


def parse_cv_form(data: dict[str, Any]) -> CvForm:
    """A cv-form from its form.json; any mistake stops with the field and what is wrong."""
    if not data.get("id"):
        raise FormError("'id' is missing")
    if not isinstance(data.get("sections"), dict) or not data["sections"]:
        raise FormError('"sections" is an object of letter -> section')
    sections: dict[str, Section] = {}
    seen: set[str] = set()
    for raw_letter, raw in data["sections"].items():
        letter = str(raw_letter).upper()
        if not re.fullmatch(r"[A-Z]", letter):
            raise FormError(f"sections: {raw_letter!r} is not one letter (photos are named by it)")
        if not isinstance(raw, dict) or not isinstance(raw.get("fields"), list):
            raise FormError(f'sections.{letter}: a section has a list of "fields"')
        fields = tuple(
            _field(f, f"sections.{letter}.fields[{n}]", seen) for n, f in enumerate(raw["fields"])
        )
        sections[letter] = Section(letter, str(raw.get("title", letter)), fields)
    pages = tuple(
        "".join(str(x) for x in page).upper() for page in data.get("pages", [list(sections)])
    )
    for n, page in enumerate(pages, 1):
        for letter in page:
            if letter not in sections:
                raise FormError(f"pages: section {letter!r} on page {n} is not in the form")
    printed = "".join(pages)
    twice = sorted({letter for letter in printed if printed.count(letter) > 1})
    if twice:
        raise FormError(f"pages: section {twice[0]!r} is on more than one page")
    prompts = data.get("prompts") or {}
    return CvForm(
        id=data["id"],
        name=data.get("name", data["id"]),
        sections=sections,
        pages=pages,
        section_prompt=prompts.get("section") or f"tally-{data['id']}",
        page_prompt=prompts.get("page") or f"tally-{data['id']}-page",
    )
