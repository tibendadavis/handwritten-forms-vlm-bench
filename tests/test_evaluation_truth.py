import json

import pytest

from formbench.evaluation.truth import TruthError, column, load_truth, normalize, truth_file


@pytest.mark.parametrize(
    "raw, key",
    [("bcg", "bcg"), ("I", "bcg"), ("I_bcg", "bcg"), ("bcg (I)", "bcg"), ("I BCG", "bcg"),
     ("L - OPV 1", "opv1"), ("Pentavalent 1", "penta1"), ("penta_1", "penta1"),
     ("A", "serial"), ("Serial Number", "serial"), ("D", "card_number"), ("Card No", "card_number"),
     ("G", "sex"), ("Date of birth", "birth_date"), ("C_birth_date", "birth_date"),
     ("age", "age"), ("Y", "mcv2"), ("J", "polio")],
)  # fmt: skip
def test_a_column_can_be_named_by_key_letter_or_both(raw, key):
    assert column(raw) == key


@pytest.mark.parametrize("raw", ["E", "child_name", "Mother/Guardian", "F phone", "H", "village", "AB"])  # fmt: skip
def test_columns_the_model_does_not_read_are_ignored(raw):
    assert column(raw) is None


def test_mistakes_in_column_names_are_explained():
    with pytest.raises(TruthError, match="birth_date and age"):
        column("C")
    with pytest.raises(TruthError, match="column K is not opv1"):
        column("K_opv1")
    with pytest.raises(TruthError, match="unknown column 'measles'"):
        column("measles")


def test_every_shape_becomes_a_page_reading():
    rows = [{"A": 1, "I": "03-06-26", "E": "a name", "K": ""}]
    for data in (rows, {"rows": rows}, {"1": {"I": "03-06-26"}}):
        page = normalize(data)
        assert page["rows"][0]["serial"] == "1" and page["rows"][0]["bcg"] == "03-06-26"
        assert page["rows"][0]["opv0"] is None and "cohort_month" not in page
    page = normalize({"Month": "June", "year": 2026, "rows": rows})
    assert page["cohort_month"] == "June" and page["cohort_year"] == "2026"


def test_the_truth_file_sits_next_to_its_photo(tmp_path):
    photo = tmp_path / "page.jpg"
    photo.touch()
    assert truth_file(photo) is None
    (tmp_path / "page.json").write_text(json.dumps([{"serial": "1", "bogus": "x"}]))
    assert truth_file(photo) == tmp_path / "page.json"
    with pytest.raises(SystemExit, match="row 1: unknown column 'bogus'"):
        load_truth(truth_file(photo))
