import json
from pathlib import Path

import pytest

from formbench.forms import load_form, parse_form
from formbench.forms.base import FormError
from formbench.forms.register import parse_register

ROOT = Path(__file__).parent.parent
REGISTER_JSON = ROOT / "tests" / "fixtures" / "register" / "form.json"


@pytest.fixture(scope="module")
def form():
    return load_form(ROOT, REGISTER_JSON)


def description():
    return json.loads(REGISTER_JSON.read_text())


REAL_STYLE_KEYS = {
    "Serial_number_A": "serial",
    "Birth_date_C": "birth_date",
    "Age_C": "age",
    "Child_card_number_D": "card_number",
    "Sex_G": "sex",
    "BCG_I": "bcg",
    "Polio_J": "polio",
    "OPV_0_K": "opv0",
    "Pentavalent_1_O": "penta1",
    "Rotavirus_dose_1_W": "rota1",
    "MCV_1_X": "mcv1",
    "Child_full_name_E": None,
    "Mobile_number_F": None,
    "Mother_guardian_name_F": None,
    "Date_of_first_visit_B": None,
    "Village_town_idp_camp_H": None,
    "Vitamin_A_12_59_months_AA": None,
    "Adverse_event_after_immunisation_AC": None,
}


def test_form_json_is_a_register(form):
    assert (form.kind, form.id, form.name, form.prompt_name) == (
        "register",
        "register",
        "Immunisation register",
        "register-page",
    )
    assert len(form.columns) == 22 and form.keys[:5] == (
        "serial",
        "card_number",
        "sex",
        "birth_date",
        "age",
    )


@pytest.mark.parametrize("raw, key", REAL_STYLE_KEYS.items())
def test_real_style_truth_keys(form, raw, key):
    truth = form.normalize_truth([{raw: "x"}])
    found = [k for k, v in truth["rows"][0].items() if v == "x"]
    assert found == ([key] if key else [])


@pytest.mark.parametrize(
    "raw, key",
    [
        ("bcg", "bcg"),
        ("I", "bcg"),
        ("i", "bcg"),
        ("I_bcg", "bcg"),
        ("bcg (I)", "bcg"),
        ("I BCG", "bcg"),
        ("Serial No", "serial"),
        ("dob", "birth_date"),
        ("Gender", "sex"),
        ("OPV 0 (birth)", "opv0"),
        ("A", "serial"),
    ],
)
def test_other_ways_to_name_a_column(form, raw, key):
    row = form.normalize_truth([{raw: "1"}])["rows"][0]
    assert row[key] == "1"


def test_plain_c_errors(form):
    with pytest.raises(FormError, match="row 1: 'C': column C holds two values"):
        form.normalize_truth([{"C": "01-01-26"}])


def test_letter_mismatch_errors(form):
    with pytest.raises(FormError, match="'bcg_J': column J is not bcg \\(that is column I\\)"):
        form.normalize_truth([{"serial": "1"}, {"bcg_J": "01-01-26"}])


def test_unknown_column_errors_with_row_and_columns(form):
    with pytest.raises(FormError) as error:
        form.normalize_truth([{"serial": "1"}, {"serial": "2", "colour": "red"}])
    message = str(error.value)
    assert message.startswith(
        "row 2: unknown column 'colour'. Columns: serial (A), card_number (D)"
    )
    assert "mcv2 (Y)" in message


def test_personal_keys_are_dropped(form):
    row = {
        "serial": "1",
        "guardian": "redacted",
        "Phone": "000",
        "mother name": "redacted",
        "Village": "North",
        "contact_address": "South",
    }
    truth = form.normalize_truth([row])
    assert set(truth["rows"][0]) == set(form.keys)
    assert "redacted" not in json.dumps(truth) and "North" not in json.dumps(truth)


def test_three_truth_shapes(form):
    rows = [{"serial": "1", "bcg": "01-06-26"}, {"serial": "2", "sex": " F "}]
    with_header = form.normalize_truth({"month": "June", "Cohort Year": "2026", "rows": rows})
    assert (with_header["cohort_month"], with_header["cohort_year"]) == ("June", "2026")
    only_rows = form.normalize_truth(rows)
    by_serial = form.normalize_truth({"1": {"bcg": "01-06-26"}, "2": {"sex": "F"}})
    assert with_header["rows"] == only_rows["rows"] == by_serial["rows"]
    assert "cohort_month" not in only_rows and "cohort_month" not in by_serial
    assert only_rows["rows"][1]["sex"] == "F" and only_rows["rows"][1]["bcg"] is None


@pytest.mark.parametrize("value", [None, "", "   "])
def test_blank_truth_values_become_none(form, value):
    assert form.normalize_truth([{"serial": "1", "bcg": value}])["rows"][0]["bcg"] is None


def test_truth_file_errors_name_the_file(form, tmp_path):
    bad = tmp_path / "page-01.json"
    bad.write_text('[{"serial": 0001}]')
    with pytest.raises(FormError) as error:
        form.load_truth(bad)
    assert str(error.value).startswith(f"{bad}: not valid JSON at line 1, column")
    assert str(error.value).endswith('Numbers with leading zeros need quotes: "0001".')
    unknown = tmp_path / "page-02.json"
    unknown.write_text('[{"colour": "red"}]')
    with pytest.raises(FormError, match=f"^{unknown}: row 1: unknown column"):
        form.load_truth(unknown)
    assert len(form.check_truths(tmp_path)) == 2


def test_truth_json_is_preferred_and_photos_without_truth_are_skipped(form, tmp_path):
    from tests.conftest import tiny_image

    folder = tmp_path / "pages"
    tiny_image(folder / "page-01.jpg")
    tiny_image(folder / "page-02.PNG")
    (folder / "page-01.json").write_text('[{"serial": "1"}]')
    (folder / "page-01.truth.json").write_text('[{"serial": "2"}]')
    items, skipped = form.items(folder, pages=False)
    assert items == [
        {
            "id": "pages/page-01.jpg",
            "input": {"photo": "page-01.jpg", "kind": "page"},
            "expected_output": form.normalize_truth([{"serial": "2"}]),
            "metadata": {"kind": "page"},
        }
    ]
    assert skipped == ["page-02.PNG (no truth file)"]
    with pytest.raises(FormError, match="--pages"):
        form.items(folder, pages=True)


def test_clean_rows(form):
    rows = [
        "not a row",
        {"serial": " 1 ", "bcg": "null", "sex": "None", "name": "redacted", "age": 3},
        {"serial": "", "bcg": None},
        {"serial": "NULL"},
    ]
    cleaned = form.clean_rows(rows)
    assert len(cleaned) == 1 and list(cleaned[0]) == list(form.keys)
    assert (cleaned[0]["serial"], cleaned[0]["bcg"], cleaned[0]["sex"], cleaned[0]["age"]) == (
        "1",
        None,
        None,
        "3",
    )
    assert form.clean_rows({"serial": "1"}) == [] and form.clean_rows(None) == []


def test_schema_has_no_personal_fields_and_keeps_key_order(form):
    schema = form.schema()
    assert list(schema["properties"]) == ["cohort_month", "cohort_year", "rows"]
    assert schema["required"] == ["cohort_month", "cohort_year", "rows"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["cohort_month"] == {
        "type": ["string", "null"],
        "description": "Month written at the top of the page",
    }
    row = schema["properties"]["rows"]["items"]
    keys = list(row["properties"])
    assert keys == [c["key"] for c in description()["columns"]]
    assert row["required"] == keys and row["additionalProperties"] is False
    assert row["properties"]["age"]["description"].startswith("C Age, bottom line")
    for word in ("name", "guardian", "phone", "village", "mother"):
        assert not any(word in k for k in keys)


def row(serial, **values):
    return {"serial": serial, **values}


def test_rows_are_paired_by_content(form):
    """One missing truth row and one invented row do not shift the rows around them."""
    truth = form.normalize_truth(
        [
            row("1", sex="F", birth_date="01-06-26", bcg="02-06-26"),
            row("2", sex="M", birth_date="03-06-26", bcg="04-06-26"),
            row("3", sex="F", birth_date="05-06-26", bcg="06-06-26"),
        ]
    )
    reading = {
        "rows": form.clean_rows(
            [
                row("77", card_number="5555", sex="M", age="3/12"),
                row("1", sex="F", birth_date="1/6/26", bcg="02-06-26"),
                row("3", sex="f", birth_date="05-06-2026", bcg="6.6.26"),
            ]
        )
    }
    counts, outcomes = form.score({}, reading, truth, True)
    by_serial = {}
    for o in outcomes:
        by_serial.setdefault(o["result"], []).append(o["field"])
    assert counts == {
        "pages": 1,
        "json_valid": 1.0,
        "field_cells": 14,
        "field_correct": 9,
        "field_tp": 9,
        "field_fp": 3,
        "field_fn": 3,
        "rows": 3,
        "rows_found": 2,
    }
    assert sorted(by_serial["misread"]) == ["serial"]
    assert sorted(by_serial["missed"]) == ["bcg", "birth_date"]
    assert sorted(by_serial["invented"]) == ["age", "card_number"]


def test_unpaired_rows_on_either_side(form):
    truth = form.normalize_truth([row("1", bcg="02-06-26"), row("2", bcg="04-06-26")])
    invented = {
        "rows": form.clean_rows(
            [row("1", bcg="02-06-26"), row("2", bcg="04-06-26"), row("9", opv1="01-07-26")]
        )
    }
    counts, outcomes = form.score({}, invented, truth, True)
    assert (counts["field_cells"], counts["field_correct"], counts["field_fp"]) == (6, 4, 2)
    assert [o["result"] for o in outcomes[-2:]] == ["invented", "invented"]
    missing = {"rows": []}
    counts, outcomes = form.score({}, missing, truth, False)
    assert (counts["field_cells"], counts["field_fn"], counts["rows_found"]) == (4, 4, 0)
    assert counts["json_valid"] == 0.0
    assert {o["result"] for o in outcomes} == {"missed"}


def test_header_cells_scored_only_when_in_truth(form):
    rows = [row("1")]
    reading = {"cohort_month": "6", "cohort_year": "26", "rows": form.clean_rows(rows)}
    counts, outcomes = form.score({}, reading, form.normalize_truth(rows), True)
    assert counts["field_cells"] == 1
    truth = form.normalize_truth({"cohort_month": "June", "cohort_year": "2026", "rows": rows})
    counts, outcomes = form.score({}, reading, truth, True)
    assert (counts["field_cells"], counts["field_correct"]) == (3, 3)
    assert [(o["group"], o["type"]) for o in outcomes[:2]] == [("header", "header")] * 2
    wrong_year = {**reading, "cohort_year": "25"}
    counts, _ = form.score({}, wrong_year, truth, True)
    assert counts["field_correct"] == 1  # a wrong year makes both header cells wrong


def test_rows_detected_counts_truth_serials_found(form):
    truth = form.normalize_truth([row("1"), row("2"), row(""), row("4")])
    reading = {"rows": form.clean_rows([row("1"), row("4 "), row("5")])}
    counts, _ = form.score({}, reading, truth, True)
    assert (counts["rows"], counts["rows_found"]) == (4, 2)


def test_empty_reading(form):
    truth = form.normalize_truth([row("1", bcg="01-01-26")])
    blank = {"cohort_month": None, "cohort_year": None, "rows": []}
    assert form.is_empty_reading({}, blank, truth) == ["page"]
    assert form.is_empty_reading({}, {**blank, "rows": truth["rows"]}, truth) == []
    assert form.is_empty_reading({}, blank, {"rows": []}) == []


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda d: d["columns"].append(dict(d["columns"][0])),
            "key 'serial' is used more than once",
        ),
        (lambda d: d["columns"][1].update(letter="G"), "letter G is also"),
        (lambda d: d["columns"][0].update(type="number"), "type 'number' is not one of"),
        (lambda d: d["columns"].pop(0), "key 'serial' is needed"),
        (lambda d: d["columns"][2].pop("description"), "columns\\[2\\]: 'description' is missing"),
        (lambda d: d["header"].append({"key": "facility"}), "header\\[2\\]: key 'facility'"),
    ],
)
def test_form_validation(change, message):
    data = description()
    change(data)
    with pytest.raises(FormError, match=message):
        parse_register(data)


def test_kind_from_shape_or_name(tmp_path):
    data = description()
    data.pop("kind")
    assert parse_form(data).kind == "register"
    with pytest.raises(FormError, match="kind 'table' is not one of"):
        parse_form({**data, "kind": "table"})
    with pytest.raises(FormError, match='has "columns"'):
        parse_form({"id": "x"})


def test_form_json_is_found_in_folder_then_parent(tmp_path):
    folder = tmp_path / "register" / "pages"
    folder.mkdir(parents=True)
    with pytest.raises(FormError, match="no form.json found: looked in"):
        load_form(folder)
    (tmp_path / "register" / "form.json").write_text(REGISTER_JSON.read_text())
    assert load_form(folder).kind == "register"
    (folder / "form.json").write_text('{"id": "x"}')
    with pytest.raises(FormError, match=f"^{folder / 'form.json'}: "):
        load_form(folder)
