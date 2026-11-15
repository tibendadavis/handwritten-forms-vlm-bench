"""What every kind of form provides to the runner, and the helpers the kinds share.

The runner knows a form only through the FormKind protocol, so adding a kind of form never
touches the benchmark flow.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from formbench.model import DEFAULT_MAX_SIDE

PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png")
LEADING_ZEROS_HINT = 'Numbers with leading zeros need quotes: "0001".'


class FormError(ValueError):
    """A form description or truth file that cannot be used, with what is wrong and where."""


@dataclass(frozen=True)
class ReadOptions:
    """How photos are read in a run."""

    think: bool = False
    max_side: int = DEFAULT_MAX_SIDE


class FormKind(Protocol):
    kind: str  # "register" or "cv-form"
    id: str  # from form.json
    name: str

    def items(self, folder: Path, pages: bool) -> tuple[list[dict], list[str]]:
        """Dataset items for the photos in `folder` that have a truth, and the photos left
        out, each with the reason."""
        ...

    def prompt_spec(self, pages: bool) -> tuple[str, str]:
        """(Langfuse prompt name, default prompt text in code)."""
        ...

    def read(
        self, folder: Path, item_input: dict, model: str, prompt: Any, options: ReadOptions
    ) -> dict:
        """Read one photo: {"reading", "json_valid", "errors", "raw"}."""
        ...

    def score(
        self, item_input: dict, reading: dict, expected: dict, json_valid: bool
    ) -> tuple[dict[str, float], list[dict]]:
        """Counts and per-field outcomes for one photo."""
        ...

    def is_empty_reading(self, item_input: dict, reading: dict, expected: dict) -> list[str]:
        """Parts (sections, or "page") that came back completely blank while the truth has
        values there."""
        ...

    def check_truths(self, folder: Path) -> list[str]:
        """Load every truth file in `folder`: one message per file that cannot be used."""
        ...


def norm(text: Any) -> str:
    """Lowercase, letters and digits only: how names are matched loosely."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def photos_in(folder: Path) -> list[Path]:
    """The photos of a folder, by name. The suffix is only a name: the real format is read
    from the file when it is sent."""
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)


def truth_files(folder: Path) -> list[Path]:
    """Every truth file in a photo folder: the JSON files other than form.json."""
    return sorted(p for p in folder.glob("*.json") if p.name != "form.json")


def find_truth(folder: Path, stem: str) -> Path | None:
    """<stem>.truth.json or <stem>.json, preferring .truth.json when both exist."""
    for name in (f"{stem}.truth.json", f"{stem}.json"):
        if (folder / name).exists():
            return folder / name
    return None


def read_json_file(path: Path) -> Any:
    """A JSON file's content; invalid JSON stops with the line, column and a hint."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FormError(
            f"{path}: not valid JSON at line {error.lineno}, column {error.colno} "
            f"({error.msg}). {LEADING_ZEROS_HINT}"
        ) from None


def outcome(
    field: str,
    group: str,
    field_type: str,
    truth: Any,
    read: Any,
    *,
    correct: bool,
    truth_blank: bool,
    read_blank: bool,
) -> dict[str, Any]:
    """One scored cell for the results file: right, misread, missed or invented."""
    if correct:
        result = "right"
    elif truth_blank:
        result = "invented"
    elif read_blank:
        result = "missed"
    else:
        result = "misread"
    return {
        "group": group,
        "field": field,
        "type": field_type,
        "truth": truth,
        "read": read,
        "result": result,
    }


def empty_counts(json_valid: bool) -> dict[str, float]:
    """The counts one photo starts from."""
    return {
        "pages": 1,
        "json_valid": float(json_valid),
        "field_cells": 0,
        "field_correct": 0,
        "field_tp": 0,
        "field_fp": 0,
        "field_fn": 0,
    }


def add_cell(counts: dict[str, float], correct: bool, truth_blank: bool, read_blank: bool) -> None:
    """Count one cell that is not blank on both sides. A misread is both a false positive (a
    wrong value reported) and a false negative (the written value not captured)."""
    counts["field_cells"] += 1
    counts["field_correct"] += correct
    counts["field_tp"] += correct and not read_blank
    counts["field_fp"] += not read_blank and not correct
    counts["field_fn"] += not truth_blank and not correct
