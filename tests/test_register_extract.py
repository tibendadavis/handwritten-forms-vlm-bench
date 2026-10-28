import cv2
import numpy as np
import pytest
from test_register_template import draw_register

from formbench.register_extract import (
    GAP,
    KEY_HEADER_H,
    ROWS_PER_TILE,
    TILE_COLUMNS,
    AlignmentFailed,
    align_to_template,
    clean_rows,
    grid_overlap,
    tiles,
)
from formbench.register_template import detect_grid


@pytest.fixture(scope="module")
def template():
    """Drawn grid plus printed-looking text, which is what features lock onto."""
    page, _, _ = draw_register()
    rng = np.random.default_rng(0)
    for i in range(90):
        x, y = int(rng.integers(120, 2700)), int(rng.integers(200, 440))
        word = "".join(chr(int(c)) for c in rng.integers(65, 91, 4))
        cv2.putText(page, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2 + (i % 3) * 0.4, 0, 3)
    cv2.putText(page, "REGISTER R-06  Month  Year", (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 6)
    return page, detect_grid(page)


def photograph(page):
    h, w = page.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[120, 180], [2700, 90], [2820, 2050], [80, 1960]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(page, matrix, (2950, 2150), borderValue=70)


def test_alignment_recovers_the_page(template):
    page, layout = template
    aligned = align_to_template(photograph(page), page)
    found = detect_grid(aligned)
    assert all(abs(a[0] - b[0]) <= 3 for a, b in zip(found["columns"].values(), layout["columns"].values()))  # fmt: skip
    assert all(abs(a["y0"] - b["y0"]) <= 3 for a, b in zip(found["rows"], layout["rows"]))


def test_wrong_lock_on_a_regular_grid_is_caught(template):
    page, _ = template
    assert grid_overlap(page, page) > 0.99
    for dx in (40, 95, 180):  # locked onto the wrong column
        assert grid_overlap(np.roll(page, dx, axis=1), page) < 0.85


def test_unrelated_image_fails_alignment(template):
    page, _ = template
    noise = np.random.default_rng(1).integers(0, 255, (800, 1000), dtype=np.uint8)
    with pytest.raises(AlignmentFailed):
        align_to_template(noise, page)


def test_tiles_keep_only_needed_columns_under_the_header(template):
    page, layout = template
    strips = tiles(page, layout)
    assert [rows for _, rows in strips][0] == list(range(ROWS_PER_TILE))
    assert sum(len(rows) for _, rows in strips) == 17
    width = sum(layout["columns"][c][1] - layout["columns"][c][0] for c in TILE_COLUMNS)
    width += GAP * (len(TILE_COLUMNS) - 1)
    first = layout["rows"][0]["y0"], layout["rows"][ROWS_PER_TILE - 1]["y1"]
    assert strips[0][0].size == (width, KEY_HEADER_H + first[1] - first[0])
    assert not {"B", "E", "F", "H"} & set(TILE_COLUMNS)


def test_clean_rows_drops_extras_and_empty_rows():
    rows = clean_rows(
        [
            {"serial": 3, "opv1": " 12-06-25 ", "child_name": "should not survive"},
            {"serial": None, "opv1": ""},
            "not a row",
        ]
    )
    assert len(rows) == 1
    assert rows[0]["serial"] == "3" and rows[0]["opv1"] == "12-06-25"
    assert "child_name" not in rows[0]
    assert clean_rows(None) == []


def test_a_timed_out_call_is_an_invalid_reading_not_a_crash(monkeypatch):
    import httpx
    import openai

    import formbench.vlm as vlm
    from formbench.register_extract import _call
    from formbench.register_schema import page_schema

    def stuck(*args, **kwargs):
        raise openai.APITimeoutError(request=httpx.Request("POST", "http://localhost"))

    monkeypatch.setattr(vlm, "complete_json", stuck)
    result = _call(None, page_schema(), "prompt", "model", False, 100, "test")
    assert result["json_valid"] is False and result["values"]["rows"] is None
    assert "timed out" in result["errors"][0]


def test_clean_rows_handles_a_failed_reading():
    assert clean_rows(None) == []
