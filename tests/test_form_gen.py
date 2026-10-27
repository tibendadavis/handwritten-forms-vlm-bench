import json
from itertools import combinations
from pathlib import Path

import pytest
import yaml

from formbench.form_gen import PAGE_H, PAGE_W, FormError, make_form, select_elements

CHILD_HEALTH = Path(__file__).parents[1] / "schemas" / "child_health"


@pytest.fixture(scope="module")
def form(tmp_path_factory):
    return make_form(CHILD_HEALTH, seed=7, out=tmp_path_factory.mktemp("forms"))


def load(folder):
    layout = json.loads((folder / "layout.json").read_text())
    schema = json.loads((folder / "extract.schema.json").read_text())
    mapping = yaml.safe_load((folder / "mapping.yaml").read_text())
    return layout, schema, mapping


def overlaps(a, b):
    return (
        a["x"] < b["x"] + b["w"]
        and b["x"] < a["x"] + a["w"]
        and a["y"] < b["y"] + b["h"]
        and b["y"] < a["y"] + a["h"]
    )


def test_writes_every_file(form):
    names = {"form.png", "form.pdf", "layout.json", "extract.schema.json", "mapping.yaml"}
    assert names <= {p.name for p in form.iterdir()}
    assert form.name == "child_health-7"


def test_every_field_has_a_cell_on_the_page(form):
    layout, _, mapping = load(form)
    assert list(layout["fields"]) == list(mapping["fields"])
    assert len(mapping["fields"]) == 6 * 4
    for box in [*layout["fields"].values(), *layout["header"].values()]:
        assert box["x"] >= 0 and box["x"] + box["w"] <= PAGE_W
        assert box["y"] >= 0 and box["y"] + box["h"] <= PAGE_H


def test_cells_do_not_overlap(form):
    layout, _, _ = load(form)
    boxes = [*layout["fields"].values(), *layout["header"].values(), *layout["markers"]]
    assert not [pair for pair in combinations(boxes, 2) if overlaps(*pair)]


def test_subset_schema_mapping_and_layout_agree(form):
    layout, schema, mapping = load(form)
    header = set(schema["properties"]) - set(mapping["fields"])
    assert header == set(layout["header"]) == {"period", "org_unit_name"}
    assert set(schema["required"]) == set(schema["properties"])


def test_same_seed_same_form(tmp_path):
    a = make_form(CHILD_HEALTH, seed=3, out=tmp_path / "a")
    b = make_form(CHILD_HEALTH, seed=3, out=tmp_path / "b")
    for name in ("form.png", "layout.json", "extract.schema.json", "mapping.yaml"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_seed_changes_selection_but_keeps_form_order():
    ds = json.loads((CHILD_HEALTH / "metadata.json").read_text())
    first, second = select_elements(ds, 6, None, 1), select_elements(ds, 6, None, 2)
    assert {f.data_element for f in first} != {f.data_element for f in second}
    order = [f.key for f in select_elements(ds, 99, None, 1)]
    assert [f.key for f in first] == [k for k in order if k in {f.key for f in first}]


def test_section_filter():
    ds = json.loads((CHILD_HEALTH / "metadata.json").read_text())
    assert {f.section for f in select_elements(ds, 3, "Nutrition", 1)} == {"Nutrition"}
    with pytest.raises(FormError, match="Immunization"):
        select_elements(ds, 3, "Malaria", 1)


def test_too_many_elements_suggests_smaller_selection(tmp_path):
    with pytest.raises(FormError, match=r"--elements \d+"):
        make_form(CHILD_HEALTH, elements=31, out=tmp_path)
