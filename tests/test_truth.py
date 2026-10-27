import json
import random
import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from formbench.form_gen import make_form
from formbench.truth import MAX_COUNT, draw_values_sheet, make_period, make_truth

CHILD_HEALTH = Path(__file__).parents[1] / "schemas" / "child_health"


def count_fields(n):
    keys = [f"f{i}" for i in range(n)]
    schema = {"properties": {"period": {}, "org_unit_name": {}, **{k: {} for k in keys}}}
    mapping = {
        "dataSet": {"periodType": "Monthly"},
        "fields": {k: {"valueType": "INTEGER", "label": k} for k in keys},
    }
    return schema, mapping


def test_values_are_counts_or_blank():
    schema, mapping = count_fields(50)
    truth = make_truth(schema, mapping, seed=1, blank_ratio=0.25)
    values = [truth[k] for k in mapping["fields"]]
    assert all(v is None or (isinstance(v, int) and 0 <= v <= MAX_COUNT) for v in values)
    assert isinstance(truth["org_unit_name"], str) and truth["org_unit_name"]


def test_blank_ratio_roughly_respected():
    schema, mapping = count_fields(2000)
    for ratio in (0.0, 0.25, 0.5, 1.0):
        truth = make_truth(schema, mapping, seed=3, blank_ratio=ratio)
        blanks = sum(truth[k] is None for k in mapping["fields"]) / 2000
        assert abs(blanks - ratio) < 0.04


def test_same_seed_same_truth():
    schema, mapping = count_fields(20)
    assert make_truth(schema, mapping, 5, 0.25) == make_truth(schema, mapping, 5, 0.25)
    assert make_truth(schema, mapping, 5, 0.25) != make_truth(schema, mapping, 6, 0.25)


@pytest.mark.parametrize(
    "period_type, pattern",
    [
        ("Daily", r"20\d\d(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])"),
        ("Weekly", r"20\d\dW([1-9]|[1-4]\d|5[0-2])"),
        ("Monthly", r"20\d\d(0[1-9]|1[0-2])"),
        ("Quarterly", r"20\d\dQ[1-4]"),
        ("Yearly", r"20\d\d"),
    ],
)
def test_period_matches_period_type(period_type, pattern):
    rng = random.Random(0)
    for _ in range(50):
        assert re.fullmatch(pattern, make_period(period_type, rng))


def test_unsupported_period_type_fails_loudly():
    with pytest.raises(ValueError, match="BiMonthly"):
        make_period("BiMonthly", random.Random(0))


def test_generated_truth_is_valid_against_the_form_schema(tmp_path):
    form = make_form(CHILD_HEALTH, seed=4, out=tmp_path)
    truth = json.loads((form / "truth.json").read_text())
    schema = json.loads((form / "extract.schema.json").read_text())
    Draft202012Validator(schema).validate(truth)
    assert (form / "values_sheet.pdf").exists()


def test_values_sheet_spills_onto_more_pages():
    schema, mapping = count_fields(120)
    truth = make_truth(schema, mapping, seed=1, blank_ratio=0.25)
    assert len(draw_values_sheet(truth, mapping, "Test")) == 3
