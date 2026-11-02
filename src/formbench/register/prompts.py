from __future__ import annotations

from typing import Any

from formbench.register.spec import DOSE_COLUMNS

# Prompts are versioned in Langfuse prompt management under these names. The text below is
# only the first version: after that, edit the prompt in Langfuse and pick a version per run.
PROMPT_NAMES = {"page": "register-page", "cover": "register-cover"}

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


def default_prompt(kind: str) -> str:
    return page_prompt() if kind == "page" else cover_prompt()


def load_prompt(langfuse: Any, kind: str, choice: str | None = None) -> Any:
    """The page or cover prompt from Langfuse: `choice` is a version number or a label
    (default: the version labelled production). The first time, the prompt in code is saved
    as version 1 with the production label. Returns None when Langfuse is not reachable and
    no version was asked for: the prompt in code is used, linked to no version."""
    from langfuse.api import NotFoundError

    name = PROMPT_NAMES[kind]
    version = int(choice) if choice and choice.isdigit() else None
    label = None if version else choice or "production"
    try:
        return langfuse.get_prompt(name, version=version, label=label, cache_ttl_seconds=0, max_retries=0)  # fmt: skip
    except NotFoundError:
        pass
    except Exception as error:  # no keys, or the server is down
        if choice:
            raise SystemExit(f"Langfuse not reachable for prompt {name} {choice} ({error})") from None  # fmt: skip
        return None
    try:
        langfuse.get_prompt(name, label="latest", cache_ttl_seconds=0, max_retries=0)
    except NotFoundError:
        if not choice:  # never saved: the prompt in code becomes version 1
            return langfuse.create_prompt(
                name=name, prompt=default_prompt(kind), labels=["production"], tags=["formbench"],
                commit_message="The prompt in code (formbench), first version",
            )  # fmt: skip
    # What was asked for does not exist: never run another version instead.
    raise SystemExit(f"Langfuse has no {name} prompt {label or version!r}. Prompts > {name} in "
                     "Langfuse lists its versions and labels.")  # fmt: skip
