import copy

import pytest

from formbench.evaluation.scoring import SCORES, metrics, score_page, sum_counts
from formbench.register.spec import load_spec

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
# 16 written cells: month, year, 8 in row 1, 6 in row 2.


def scores(predicted, truth=TRUTH, json_valid=True):
    return metrics(score_page(predicted, truth, json_valid))


def test_perfect_page_scores_perfectly():
    s = scores(copy.deepcopy(TRUTH))
    assert s == {"cells_correct": 1, "written_cells_found": 1, "output_values_right": 1,
                 "rows_found": 1, "valid_output": 1}  # fmt: skip


def test_formatting_differences_are_not_errors():
    pred = copy.deepcopy(TRUTH)
    pred["cohort_month"], pred["cohort_year"] = "06", "26"
    pred["rows"][0]["opv1"] = "17\n07\n26"  # stacked in a narrow column
    pred["rows"][0]["birth_date"] = "3/6/26"
    pred["rows"][1]["sex"] = "female"
    assert scores(pred)["cells_correct"] == 1


def test_a_value_under_the_wrong_column_counts_twice():
    pred = copy.deepcopy(TRUTH)
    pred["rows"][0]["penta1"], pred["rows"][0]["pcv1"] = None, "17-07-26"
    s = scores(pred)
    # penta1 missed and pcv1 invented: 15 right of 17 cells
    assert s["cells_correct"] == pytest.approx(15 / 17, abs=1e-4)
    assert s["written_cells_found"] == s["output_values_right"] == pytest.approx(15 / 16, abs=1e-4)


def test_missed_values_lower_found_and_invented_values_lower_right():
    pred = copy.deepcopy(TRUTH)
    pred["rows"][1]["opv0"] = None
    s = scores(pred)
    assert s["written_cells_found"] == pytest.approx(15 / 16, abs=1e-4)
    assert s["output_values_right"] == 1
    pred = copy.deepcopy(TRUTH)
    pred["rows"][1]["opv1"] = "12-07-26"
    s = scores(pred)
    assert s["written_cells_found"] == 1
    assert s["output_values_right"] == pytest.approx(16 / 17, abs=1e-4)


def test_invented_and_missing_rows():
    pred = copy.deepcopy(TRUTH)
    pred["rows"].append({"serial": "3", "sex": "F", "opv0": "11-06-26"})
    s = scores(pred)
    assert s["cells_correct"] == pytest.approx(16 / 19, abs=1e-4) and s["rows_found"] == 1
    pred = copy.deepcopy(TRUTH)
    pred["rows"].pop()
    s = scores(pred)
    assert s["cells_correct"] == pytest.approx(10 / 16, abs=1e-4) and s["rows_found"] == 0.5


def test_the_cohort_counts_only_when_the_truth_has_it():
    truth = {"rows": copy.deepcopy(TRUTH["rows"])}
    assert scores(copy.deepcopy(TRUTH), truth)["cells_correct"] == 1


def test_invalid_json_shows_in_valid_output():
    pages = [score_page(copy.deepcopy(TRUTH), TRUTH, True), score_page({"rows": []}, TRUTH, False)]
    assert metrics(sum_counts(pages))["valid_output"] == 0.5


def test_eval_differences_list_wrong_cells_missing_and_invented_rows():
    from formbench.evaluation.photo import differences

    pred = copy.deepcopy(TRUTH)
    pred["rows"][0]["opv1"] = "17-01-26"
    pred["rows"][1]["serial"] = "9"
    diffs = differences(pred, TRUTH)
    assert ("1", "opv1", "17-07-26", "17-01-26") in diffs
    assert ("2", "(whole row)", "present", "missing") in diffs
    assert ("9", "(whole row)", "absent", "invented") in diffs


def test_dhis2_values(ri_metadata):
    spec = load_spec(ri_metadata)
    perfect = metrics(score_page(copy.deepcopy(TRUTH), TRUTH, True, spec))
    assert perfect["dhis2_values_right"] == 1
    pred = copy.deepcopy(TRUTH)
    pred["rows"][0]["penta1"], pred["rows"][0]["pcv1"] = None, "17-07-26"  # filed one column off
    s = metrics(score_page(pred, TRUTH, True, spec))
    assert 0 < s["dhis2_values_right"] < 1
    assert "dhis2_values_right" not in metrics(score_page(pred, TRUTH, True))
    no_cohort = {"rows": TRUTH["rows"]}  # counted with the model's cohort
    assert (
        metrics(score_page(copy.deepcopy(TRUTH), no_cohort, True, spec))["dhis2_values_right"] == 1
    )


def test_every_score_is_described_once():
    names = [s.name for s in SCORES]
    assert len(names) == len(set(names))
    assert all(s.label and s.description and s.group in ("reading", "dhis2", "cost") for s in SCORES)  # fmt: skip
