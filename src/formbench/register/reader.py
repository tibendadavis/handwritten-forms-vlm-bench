"""Reading register photos with a VLM: one call per whole page photo, one for the cover."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from formbench.prompt_store import prompt_version
from formbench.register.prompts import (
    PROMPT_NAMES,
    ROW_FIELDS,
    cover_schema,
    default_prompt,
    page_schema,
)

# Users photograph the whole (A3) page; dates in the narrow dose columns need the detail.
# 4032 px is a 12 MP phone photo as taken: larger photos are shrunk, smaller ones sent as is.
FULL_MAX_SIDE = 4032
# Some models write an empty cell as the text "null" instead of JSON null.
EMPTY = {"", "null", "none"}


def _cell(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return None if text.lower() in EMPTY else text


def clean_rows(rows: Any) -> list[dict[str, Any]]:
    """Keep only schema keys, as strings or None; drop rows with nothing written."""
    out = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        clean = {k: _cell(row.get(k)) for k in ROW_FIELDS}
        if any(clean.values()):
            out.append(clean)
    return out


def _call(image, schema, kind, prompt, model, think, max_side):
    """One model call with the Langfuse `prompt` (None: the prompt in code); a timeout is an
    invalid reply, not a crash (vlm.read_json)."""
    from formbench.vlm import read_json

    text = default_prompt(kind) if prompt is None else prompt.compile()
    return read_json(
        image, schema, text, model, think=think, max_side=max_side, name=PROMPT_NAMES[kind],
        metadata={"prompt_version": prompt_version(prompt)}, langfuse_prompt=prompt,
    )  # fmt: skip


def read_page(photo: Path, model: str, *, think: bool = False, max_side: int = FULL_MAX_SIDE,
              prompt: Any = None) -> dict[str, Any]:  # fmt: skip
    """Transcribe one cohort page photo with a Langfuse prompt (None: the prompt in code).
    Returns the reading plus validity and the raw reply."""
    result = _call(photo, page_schema(), "page", prompt, model, think, max_side)
    values = result["values"]
    reading = {
        "cohort_month": values.get("cohort_month"),
        "cohort_year": values.get("cohort_year"),
        "rows": clean_rows(values.get("rows")),
    }
    return {"reading": reading, "json_valid": result["json_valid"], "errors": result["errors"],
            "raw": result["raw"]}  # fmt: skip


def read_cover(photo: Path, model: str, *, think: bool = False, prompt: Any = None) -> dict[str, Any]:  # fmt: skip
    result = _call(photo, cover_schema(), "cover", prompt, model, think, FULL_MAX_SIDE)
    return {"cover": result["values"], "json_valid": result["json_valid"],
            "errors": result["errors"], "raw": result["raw"]}  # fmt: skip
