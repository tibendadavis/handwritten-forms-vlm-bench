import copy

import pytest

from formbench.evaluation.scoring import quality_metrics, schema_pass, score_cover, score_page

TRUTH = {
    "cohort_month": "June",
    "cohort_year": "2026",
    "rows": [
        {"serial": "1", "card_number": "7654", "sex": "M", "birth_date": "03-06-26",
         "age": "1/12", "opv0": "05-06-26", "opv1": "17-07-26", "penta1": "17-07-26"},
        {"serial": "2", "card_number": "551", "sex": "F", "birth_date": "08-06-26",
         "age": "2/7", "opv0": "10-06-26", "opv1": None, "penta1": None},
    ],
}  # fmt: skip


def scores(predicted, json_valid=True):
    return quality_metrics(score_page(predicted, TRUTH, json_valid))


def test_perfect_page_scores_perfectly():
    s = scores(copy.deepcopy(TRUTH))
    assert s["cer"] == 0 and s["wer"] == 0
    assert s["anls"] == s["anlsl"] == s["field_accuracy"] == s["field_f1"] == 1
    assert s["stp_rate"] == s["ganted"] == s["tree_f1"] == s["schema_pass_rate"] == 1


def test_formatting_differences_are_not_errors():
    pred = copy.deepcopy(TRUTH)
    pred["cohort_month"], pred["cohort_year"] = "06", "26"
    pred["rows"][0]["opv1"] = "17\n07\n26"  # stacked in a narrow column
    pred["rows"][0]["birth_date"] = "3/6/26"
    s = scores(pred)
    assert s["field_accuracy"] == 1 and s["tree_f1"] == 1 and s["stp_rate"] == 1


def test_value_in_the_wrong_column_hurts_structure_more_than_text():
    pred = copy.deepcopy(TRUTH)
    pred["rows"][0]["penta1"], pred["rows"][0]["opv1"] = None, "17-07-26"
    pred["rows"][0]["opv0"] = None
    pred["rows"][0]["polio"] = "05-06-26"  # OPV 0 date read into the Polio column
    s = scores(pred)
    assert s["field_f1"] < 1 and s["tree_f1"] < 1 and s["stp_rate"] == 0
    assert s["ganted"] < 1


def test_misread_digit_costs_little_cer_but_breaks_the_field():
    pred = copy.deepcopy(TRUTH)
    pred["rows"][0]["opv1"] = "17-01-26"
    s = scores(pred)
    assert 0 < s["cer"] < 0.05
    assert s["field_accuracy"] < 1 and s["anls"] > 0.9


def test_invented_and_missing_rows():
    pred = copy.deepcopy(TRUTH)
    pred["rows"].append({"serial": "3", "sex": "F", "opv0": "11-06-26"})
    assert scores(pred)["anlsl"] == pytest.approx(2 / 3, abs=1e-4)
    pred = copy.deepcopy(TRUTH)
    pred["rows"].pop()
    s = scores(pred)
    assert s["anlsl"] == pytest.approx(0.5, abs=1e-4) and s["stp_rate"] == 0


def test_schema_pass_needs_valid_json_and_dhis2_formats():
    assert schema_pass(TRUTH, json_valid=True)
    assert not schema_pass(TRUTH, json_valid=False)
    bad = copy.deepcopy(TRUTH)
    bad["rows"][0]["opv1"] = "31-02-26"
    assert not schema_pass(bad, json_valid=True)
    bad = copy.deepcopy(TRUTH)
    bad["rows"][1]["sex"] = "X"
    assert not schema_pass(bad, json_valid=True)


def test_cover_text_scores():
    truth = {"facility_name": "Riverside Health Centre", "mfl_code": "MFL-1234"}
    counts = score_cover({"facility_name": "Riverside Helth Centre", "mfl_code": "MFL-1234"}, truth)  # fmt: skip
    s = quality_metrics(counts)
    assert 0 < s["cer"] < 0.05 and s["cover_field_accuracy"] == 0.5


def test_eval_differences_list_wrong_cells_missing_and_invented_rows():
    from formbench.evaluation.photo import differences

    pred = copy.deepcopy(TRUTH)
    pred["rows"][0]["opv1"] = "17-01-26"
    pred["rows"][1]["serial"] = "9"
    diffs = differences(pred, TRUTH)
    assert ("1", "opv1", "17-07-26", "17-01-26") in diffs
    assert ("2", "(whole row)", "present", "missing") in diffs
    assert ("9", "(whole row)", "absent", "invented") in diffs
