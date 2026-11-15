"""Finding a folder's form.json and turning it into the right kind of form."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from formbench.forms.base import FormError, FormKind, read_json_file
from formbench.forms.cv_form import parse_cv_form
from formbench.forms.register import parse_register
from formbench.forms.summary import parse_summary

KINDS = ("register", "cv-form", "summary")


def find_form(folder: Path, explicit: Path | None = None) -> Path:
    """`explicit` when given; otherwise form.json in the photo folder, then in its parent."""
    candidates = [explicit] if explicit else [folder / "form.json", folder.parent / "form.json"]
    for path in candidates:
        if path.is_file():
            return path
    looked = " or ".join(str(p) for p in candidates)
    raise FormError(f"no form.json found: looked in {looked} (give one with --form PATH)")


def kind_of(data: dict[str, Any]) -> str:
    """The kind named in form.json, or the one its shape says: columns, sections or fields."""
    if "kind" in data:
        if data["kind"] not in KINDS:
            raise FormError(f"kind {data['kind']!r} is not one of {KINDS}")
        return data["kind"]
    if "columns" in data:
        return "register"
    if "sections" in data:
        return "cv-form"
    if "fields" in data:
        return "summary"
    raise FormError(
        'a form has "columns" (a register), "sections" (a cv-form) or "fields" (a summary)'
    )


def parse_form(data: Any) -> FormKind:
    """A form from the content of its form.json."""
    if not isinstance(data, dict):
        raise FormError("a form description is a JSON object")
    parsers = {"register": parse_register, "cv-form": parse_cv_form, "summary": parse_summary}
    return parsers[kind_of(data)](data)


def load_form(folder: Path, explicit: Path | None = None) -> FormKind:
    """The form that describes the photos in `folder`; any mistake names the file."""
    path = find_form(folder, explicit)
    data = read_json_file(path)
    try:
        return parse_form(data)
    except FormError as error:
        raise FormError(f"{path}: {error}") from None
