"""Finding a folder's form.json and turning it into the right kind of form."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from formbench.forms.base import FormError, FormKind, read_json_file
from formbench.forms.cv_form import parse_cv_form
from formbench.forms.register import parse_register

KINDS = ("register", "cv-form")


def find_form(folder: Path, explicit: Path | None = None) -> Path:
    """`explicit` when given; otherwise form.json in the photo folder, then in its parent."""
    candidates = [explicit] if explicit else [folder / "form.json", folder.parent / "form.json"]
    for path in candidates:
        if path.is_file():
            return path
    looked = " or ".join(str(p) for p in candidates)
    raise FormError(f"no form.json found: looked in {looked} (give one with --form PATH)")


def kind_of(data: dict[str, Any]) -> str:
    """The kind named in form.json, or the one its shape says: columns or sections."""
    if "kind" in data:
        if data["kind"] not in KINDS:
            raise FormError(f"kind {data['kind']!r} is not one of {KINDS}")
        return data["kind"]
    if "columns" in data:
        return "register"
    if "sections" in data:
        return "cv-form"
    raise FormError('a form has "columns" (a register) or "sections" (a cv-form)')


def parse_form(data: Any) -> FormKind:
    """A form from the content of its form.json."""
    if not isinstance(data, dict):
        raise FormError("a form description is a JSON object")
    kind = kind_of(data)
    return parse_register(data) if kind == "register" else parse_cv_form(data)


def load_form(folder: Path, explicit: Path | None = None) -> FormKind:
    """The form that describes the photos in `folder`; any mistake names the file."""
    path = find_form(folder, explicit)
    data = read_json_file(path)
    try:
        return parse_form(data)
    except FormError as error:
        raise FormError(f"{path}: {error}") from None
