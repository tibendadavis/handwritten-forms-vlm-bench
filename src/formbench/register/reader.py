"""Reading register photos with a VLM: one call per whole page photo, one for the cover."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from formbench.register.prompts import (
    REGISTER_PROMPT_VERSION,
    ROW_FIELDS,
    cover_prompt,
    cover_schema,
    page_prompt,
    page_schema,
)

# Users photograph the whole (A3) page; dates in the narrow dose columns need the detail.
FULL_MAX_SIDE = 3072


def clean_rows(rows: Any) -> list[dict[str, Any]]:
    """Keep only schema keys, as strings or None; drop rows with nothing written."""
    out = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        clean = {
            k: (str(row[k]).strip() or None) if row.get(k) not in (None, "") else None
            for k in ROW_FIELDS
        }
        if any(clean.values()):
            out.append(clean)
    return out


def _call(image, schema, prompt, model, think, max_side, name):
    """One model call. A timeout is recorded as an invalid reply (nothing read), not a crash,
    so one stuck page cannot stop a whole benchmark; other errors (e.g. Ollama down) raise."""
    import openai

    from formbench.vlm import complete_json, interpret

    try:
        raw = complete_json(
            image, schema, prompt, model, think=think, max_side=max_side, name=name,
            metadata={"prompt_version": REGISTER_PROMPT_VERSION},
        )  # fmt: skip
    except openai.APITimeoutError:
        return {"raw": None, "values": dict.fromkeys(schema["properties"]), "json_valid": False,
                "errors": ["model call timed out (no reply)"]}  # fmt: skip
    return interpret(raw, schema)


def read_page(photo: Path, model: str, *, think: bool = False, max_side: int = FULL_MAX_SIDE) -> dict[str, Any]:  # fmt: skip
    """Transcribe one cohort page photo. Returns the reading plus validity and the raw reply."""
    result = _call(photo, page_schema(), page_prompt(), model, think, max_side, "register-page")
    values = result["values"]
    reading = {
        "cohort_month": values.get("cohort_month"),
        "cohort_year": values.get("cohort_year"),
        "rows": clean_rows(values.get("rows")),
    }
    return {"reading": reading, "json_valid": result["json_valid"], "errors": result["errors"],
            "raw": result["raw"]}  # fmt: skip


def read_cover(photo: Path, model: str, *, think: bool = False) -> dict[str, Any]:
    result = _call(photo, cover_schema(), cover_prompt(), model, think, FULL_MAX_SIDE, "register-cover")  # fmt: skip
    return {"cover": result["values"], "json_valid": result["json_valid"],
            "errors": result["errors"], "raw": result["raw"]}  # fmt: skip
