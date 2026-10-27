import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from formbench.form_gen import make_form
from formbench.ocr_baseline import read_form

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs tesseract")

CHILD_HEALTH = Path(__file__).parents[1] / "schemas" / "child_health"


def fill(folder, values):
    """Write values into their boxes on form.png, like a very neat person would."""
    layout = json.loads((folder / "layout.json").read_text())
    page = Image.open(folder / "form.png")
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=72)
    boxes = {**layout["header"], **layout["fields"]}
    for key, value in values.items():
        if value is not None:
            box = boxes[key]
            centre = (box["x"] + box["w"] // 2, box["y"] + box["h"] // 2)
            draw.text(centre, str(value), font=font, fill=0, anchor="mm")
    return np.array(page), layout


def test_reads_printed_digits_and_leaves_blanks_null(tmp_path):
    folder = make_form(CHILD_HEALTH, seed=5, out=tmp_path)
    truth = json.loads((folder / "truth.json").read_text())
    page, layout = fill(folder, truth)
    result = read_form(page, layout)
    fields = layout["fields"]
    assert {k: result["values"][k] for k in fields} == {k: truth[k] for k in fields}
    assert result["values"]["period"] == truth["period"]
    assert set(result["values"]) == set(result["confidence"]) == {*fields, *layout["header"]}
    assert all(0 <= c <= 1 for c in result["confidence"].values())


def test_empty_form_reads_all_null(tmp_path):
    folder = make_form(CHILD_HEALTH, seed=5, out=tmp_path)
    page, layout = fill(folder, {})
    assert set(read_form(page, layout)["values"].values()) == {None}
