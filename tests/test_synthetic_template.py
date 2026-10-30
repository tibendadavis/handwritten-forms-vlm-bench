import numpy as np
import pytest

from formbench.synthetic.template import LETTERS, ROWS, TemplateError, detect_grid


def draw_register(rows=ROWS):
    """A blank register-like grid: header band, letter row, child rows with a C/F sub-line."""
    page = np.full((2200, 3000), 255, np.uint8)
    # Irregular widths like the real register (wide name columns, narrow dose columns).
    widths = [70 + (i * 37) % 50 for i in range(len(LETTERS))]
    xs = [100 + sum(widths[:i]) for i in range(len(LETTERS) + 1)]
    header = [150, 300, 420, 460]
    body = [460 + i * 100 for i in range(rows + 1)]
    for y in header + body:
        page[y : y + 3, xs[0] : xs[-1] + 3] = 0
    for x in xs:
        page[header[0] : body[-1] + 3, x : x + 3] = 0
    for top in body[:-1]:  # sub-line inside columns C and F only
        for col in (2, 5):
            page[top + 50 : top + 53, xs[col] : xs[col + 1]] = 0
    return page, xs, body


def test_detects_columns_rows_and_sublines():
    page, xs, body = draw_register()
    grid = detect_grid(page)
    assert list(grid["columns"]) == LETTERS
    assert grid["columns"]["A"] == [xs[0] + 1, xs[1] + 1]
    assert len(grid["rows"]) == ROWS
    assert [r["y0"] for r in grid["rows"]] == [y + 1 for y in body[:-1]]
    assert all(r["split"] == r["y0"] + 50 for r in grid["rows"])
    assert grid["header"] == {"y0": 151, "y1": body[0] + 1}


def test_wrong_page_fails_loudly():
    page, _, _ = draw_register(rows=10)
    with pytest.raises(TemplateError, match="17 child rows"):
        detect_grid(page)
