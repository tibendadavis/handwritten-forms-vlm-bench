import pytest

from formbench.scoring import normalize, score
from formbench.vlm import interpret

SCHEMA = {
    "properties": {
        "org_unit_name": {"type": ["string", "null"]},
        "a": {"type": ["integer", "null"]},
        "b": {"type": ["number", "null"]},
        "c": {"type": ["integer", "null"]},
    }
}
TRUTH = {"org_unit_name": "Amani North Clinic", "a": 12, "b": 7, "c": None}


def by_key(result):
    return {f["key"]: f for f in result["fields"]}


@pytest.mark.parametrize(
    "value, numeric, expected",
    [
        (" 12 ", True, 12),
        ("12.0", True, 12),
        (12.0, True, 12),
        ("", True, None),
        ("  ", False, None),
        ("twelve", True, "twelve"),
        ("  Amani   NORTH clinic ", False, "amani north clinic"),
        ("202501", False, "202501"),
    ],
)
def test_normalize(value, numeric, expected):
    assert normalize(value, numeric) == expected


def test_all_correct():
    pred = {"org_unit_name": "amani north  clinic", "a": "12", "b": 7.0, "c": ""}
    result = score(pred, TRUTH, SCHEMA, json_valid=True)
    assert result["metrics"] == {
        "json_valid": 1,
        "field_accuracy": 1.0,
        "numeric_accuracy": 1.0,
        "hallucination_rate": 0.0,
        "omission_rate": 0.0,
    }


def test_wrong_value():
    pred = {**TRUTH, "a": 21}
    result = score(pred, TRUTH, SCHEMA, json_valid=True)
    assert not by_key(result)["a"]["correct"]
    assert result["metrics"]["field_accuracy"] == 0.75
    assert result["metrics"]["numeric_accuracy"] == 0.5
    assert result["metrics"]["omission_rate"] == 0.0


def test_hallucination():
    result = score({**TRUTH, "c": 3}, TRUTH, SCHEMA, json_valid=True)
    assert by_key(result)["c"]["hallucinated"]
    assert result["metrics"]["hallucination_rate"] == 1.0


def test_omission():
    result = score({**TRUTH, "b": None}, TRUTH, SCHEMA, json_valid=True)
    assert by_key(result)["b"]["omitted"]
    assert result["metrics"]["omission_rate"] == pytest.approx(1 / 3)


def test_invalid_json_scores_as_all_omitted():
    reply = interpret("Sorry, I can't read that.", SCHEMA)
    result = score(reply["values"], TRUTH, SCHEMA, reply["json_valid"])
    assert result["metrics"]["json_valid"] == 0
    assert result["metrics"]["omission_rate"] == 1.0
    assert result["metrics"]["field_accuracy"] == 0.25  # only the blank field is right


def test_rate_without_denominator_is_none():
    truth = {"org_unit_name": "X", "a": 1, "b": 2, "c": 3}
    assert score(truth, truth, SCHEMA, json_valid=True)["metrics"]["hallucination_rate"] is None
