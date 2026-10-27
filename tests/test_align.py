import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from formbench.align import MarkersNotFound, align, find_markers, load_gray, marker_centres
from formbench.form_gen import make_form

CHILD_HEALTH = Path(__file__).parents[1] / "schemas" / "child_health"


@pytest.fixture(scope="module")
def form(tmp_path_factory):
    folder = make_form(CHILD_HEALTH, seed=2, out=tmp_path_factory.mktemp("forms"))
    return load_gray(folder / "form.png"), json.loads((folder / "layout.json").read_text())


def photograph(page, background=90):
    """Simulate a phone photo: page tilted in perspective on a dark table, scaled down."""
    h, w = page.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[310, 240], [1690, 330], [1780, 2380], [180, 2290]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(page, matrix, (2000, 2600), borderValue=background)


def test_markers_found_in_template_order(form):
    page, layout = form
    assert np.abs(find_markers(page) - marker_centres(layout)).max() < 1.5


def test_alignment_recovers_cell_positions(form):
    page, layout = form
    aligned = align(photograph(page), layout)
    assert aligned.shape == page.shape
    assert np.abs(find_markers(aligned) - marker_centres(layout)).max() < 4
    for box in [*layout["fields"].values(), *layout["header"].values()]:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        # Borders land where the layout says: the cell edge is dark, its centre white.
        assert aligned[y + h // 2, x : x + 4].min() < 128
        assert aligned[y + h // 2, x + w // 2] > 200


def test_colour_photo_is_accepted(form):
    page, layout = form
    colour = cv2.cvtColor(photograph(page), cv2.COLOR_GRAY2BGR)
    assert align(colour, layout).shape == page.shape


def test_missing_markers_raise_clear_error(form):
    _, layout = form
    with pytest.raises(MarkersNotFound, match="four corner markers"):
        align(np.full((1000, 800), 255, np.uint8), layout)
