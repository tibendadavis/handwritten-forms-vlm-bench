"""The summary kind: a monthly summary report, printed as tables in which totals are written by
hand, one whole page per photo.

form.json is a flat list of "fields", each written like a cv-form field ({"key", "type"}); the
page has no lettered sections. A field's key says where it is on the page, row then column (for
example "BCG_0-11 m_Facility"); shaded cells are simply not listed. Fields without a type are
counts. Breakdowns group fields by row: the key up to its first "_".

A truth file is <photo stem>.json next to the photo: field key -> value for the whole page.
Personal fields (the reporting officer) are never in the schema; in truth files they are dropped
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
from formbench.forms.cv_form import Field, slug
from formbench.prompts import SUMMARY_PAGE_PROMPT, compile_prompt, prompt_version
from formbench.values import clean_value, field_blank, same

TYPES = ("count", "digits", "text", "date", "mark")  # no circles to fill on this form
# Numbers are written freehand in a table cell here, not one digit per box.
COUNT_HINT = "number written in the cell"
YEAR_FIRST = re.compile(r"^\s*(\d{4})\s*[-/.\s]\s*(\d{1,2})\s*[-/.\s]\s*(\d{1,2})\s*$")


def day_first(value: Any) -> Any:
    """A year-first date ("2026/10/03", as this form asks) as day-month-year, so it compares
    with the shared date rules; anything else unchanged."""
    match = YEAR_FIRST.match("" if value is None else str(value))
    return f"{match[3]}-{match[2]}-{match[1]}" if match else value


def same_value(field: Field, truth: Any, read: Any) -> bool:
    """Whether a field was read right: the shared cv-form rules, dates also year first."""
    if field.type == "date":
        truth, read = day_first(truth), day_first(read)
    return same(field.type, truth, read)


def hint(field: Field) -> str:
    """What the model should return for this field, as shown in the prompt and schema."""
    return COUNT_HINT if field.type == "count" else field.hint()


def row_of(field: Field) -> str:
    """The row a field is on, its group in breakdowns: "BCG_0-11 m_Facility" -> "BCG"."""
    return field.key.split("_")[0]


def fields_variable(fields: tuple[Field, ...]) -> str:
    """The prompt's {{fields}}: one line per field, in page order."""
    return "\n".join(f"- {f.id}: {f.key.replace('_', ', ')} ({hint(f)})" for f in fields)


@dataclass(frozen=True)
class SummaryForm:
    id: str
    name: str
    prompt_name: str
    fields: tuple[Field, ...]
    personal: tuple[str, ...]
    kind: str = "summary"

    def field(self, key: str) -> Field | None:
        return next((f for f in self.fields if f.id == slug(key)), None)

    # truth files

    def load_truth(self, path: Path) -> dict[str, Any]:
        """A whole page as field id -> value, every field present. Personal fields are dropped;
        any other unknown key stops with its name."""
        data = read_json_file(path)
        if isinstance(data, list) and len(data) == 1:
            data = data[0]
        if not isinstance(data, dict):
            raise FormError(f"{path}: a truth file is one object of field -> value")
        out: dict[str, Any] = {f.id: None for f in self.fields}
        for key, value in data.items():
            field = self.field(str(key))
            if field is not None:
                out[field.id] = value
            elif not any(word in norm(key) for word in self.personal):
                raise FormError(f"{path}: unknown field {key!r}; fields are named as in form.json")
        return out

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
            raise FormError("--pages is for cv-forms; a summary is always one photo per page")
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
        """One reply object with every field, in form order."""
        props = {
            f.id: {"type": ["string", "null"], "description": f"{f.key}: {hint(f)}"}
            for f in self.fields
        }
        return {
            "type": "object",
            "properties": props,
            "required": list(props),
            "additionalProperties": False,
        }

    def prompt_spec(self, pages: bool = False) -> tuple[str, str]:
        return self.prompt_name, SUMMARY_PAGE_PROMPT

    def read(
        self, folder: Path, item_input: dict, model_tag: str, prompt: Any, options: ReadOptions
    ) -> dict:
        """One call per page; the reading is field id -> value."""
        result = model.read_json(
            folder / item_input["photo"],
            self.schema(),
            compile_prompt(prompt, {"fields": fields_variable(self.fields)}),
            model_tag,
            think=options.think,
            max_side=options.max_side,
            trace_name=self.prompt_name,
            metadata={"prompt_version": prompt_version(prompt)},
            langfuse_prompt=prompt,
        )
        values = result["values"]
        return {
            "reading": {f.id: clean_value(values.get(f.id)) for f in self.fields},
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
        for f in self.fields:
            t, r = expected.get(f.id), reading.get(f.id)
            t_blank, r_blank = field_blank(f.type, t), field_blank(f.type, r)
            if t_blank and r_blank:
                continue
            correct = same_value(f, t, r)
            add_cell(counts, correct, t_blank, r_blank)
            outcomes.append(
                outcome(
                    f.id,
                    row_of(f),
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
        read_any = any(not field_blank(f.type, reading.get(f.id)) for f in self.fields)
        truth_any = any(not field_blank(f.type, expected.get(f.id)) for f in self.fields)
        return ["page"] if truth_any and not read_any else []


def _field(raw: Any, n: int, seen: set[str]) -> Field:
    where = f"fields[{n}]"
    if not isinstance(raw, dict) or not raw.get("key"):
        raise FormError(f'{where}: a field is an object with a "key"')
    field = Field(str(raw["key"]), raw.get("type", "count"))
    where = f"{where} ({field.key})"
    if field.type not in TYPES:
        raise FormError(f"{where}: type {field.type!r} is not one of {TYPES}")
    if not field.id:
        raise FormError(f"{where}: the key needs letters or digits")
    if field.id in seen:
        raise FormError(f"{where}: another field has the same id {field.id!r}")
    seen.add(field.id)
    return field


def parse_summary(data: dict[str, Any]) -> SummaryForm:
    """A summary form from its form.json; any mistake stops with the field and what is wrong."""
    if not data.get("id"):
        raise FormError("'id' is missing")
    if not isinstance(data.get("fields"), list) or not data["fields"]:
        raise FormError('"fields" is a list of fields')
    seen: set[str] = set()
    return SummaryForm(
        id=data["id"],
        name=data.get("name", data["id"]),
        prompt_name=data.get("prompt") or f"summary-{data['id']}",
        fields=tuple(_field(raw, n, seen) for n, raw in enumerate(data["fields"])),
        personal=tuple(norm(w) for w in data.get("personal_words", [])),
    )
