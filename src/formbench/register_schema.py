from __future__ import annotations

from typing import Any

from formbench.register_spec import DOSE_COLUMNS

# Bump when a register prompt or schema changes so runs stay comparable.
REGISTER_PROMPT_VERSION = "r2"

TEXT = {"type": ["string", "null"]}

# Row fields the model may return. Names, guardian, phone and village are deliberately absent:
# with additionalProperties false they cannot come back, so they never reach traces or files.
ROW_FIELDS = {
    "serial": "A Serial Number",
    "card_number": "D Child Card Number",
    "sex": "G Sex, M or F as written",
    "birth_date": "C Birth Date, top line of the cell, exactly as written",
    "age": "C Age, bottom line of the cell, exactly as written (e.g. 7, 3/12, 6/52, 5/7)",
    **{c.key: f"{c.letter} {c.label} date, exactly as written" for c in DOSE_COLUMNS},
}


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def rows_schema() -> dict[str, Any]:
    row = _obj({k: {**TEXT, "description": d} for k, d in ROW_FIELDS.items()})
    return _obj({"rows": {"type": "array", "items": row}})


def cohort_schema() -> dict[str, Any]:
    return _obj(
        {
            "cohort_month": {**TEXT, "description": "Month written at the top of the page"},
            "cohort_year": {**TEXT, "description": "Year written at the top of the page"},
        }
    )


def page_schema() -> dict[str, Any]:
    schema = cohort_schema()
    schema["properties"].update(rows_schema()["properties"])
    schema["required"] = list(schema["properties"])
    return schema


def cover_schema() -> dict[str, Any]:
    fields = {
        "facility_name": "Facility Name",
        "mfl_code": "MFL Code",
        "facility_type": "Facility Type",
        "district": "District",
        "region": "Region",
    }
    return _obj({k: {**TEXT, "description": f"{d}, exactly as written"} for k, d in fields.items()})


_RULES = (
    "Rules:\n"
    "- Copy every value exactly as handwritten; do not reformat dates or correct anything.\n"
    "- Use null for an empty or unreadable cell. Never guess.\n"
    "- One object per child row that has anything written in it; skip empty rows.\n"
    "- Do not transcribe child names, mother/guardian names, phone numbers or villages.\n"
    "- Output only the JSON object."
)


def _columns() -> str:
    return "\n".join(f"- {k}: column {d}" for k, d in ROW_FIELDS.items())


def page_prompt() -> str:
    return (
        "This is a photo of one page of a handwritten immunisation register. "
        "Columns are labelled with letters A to AC under the header. "
        "Return cohort_month and cohort_year from the page header, and rows with these keys:\n"
        f"{_columns()}\n\n{_RULES}"
    )


def cover_prompt() -> str:
    return (
        "This is the cover of a handwritten immunisation register. Copy each field exactly as "
        "written, null if empty: Facility Name, MFL Code, Facility Type, District, Region. "
        "Output only the JSON object."
    )
