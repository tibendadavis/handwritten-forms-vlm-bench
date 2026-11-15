import json
import shutil

import pytest

from formbench.forms import load_form, parse_form
from formbench.forms.base import FormError
from formbench.forms.cv_form import (
    page_variables,
    parse_cv_form,
    schema_for,
    section_variables,
    slug,
)
from tests.conftest import FIXTURES, tiny_image

SAMPLES = FIXTURES / "cv-form" / "samples"


def description():
    return json.loads((FIXTURES / "cv-form" / "form.json").read_text())


@pytest.fixture(scope="module")
def form():
    return load_form(SAMPLES)


def test_form_is_found_in_the_parent_folder(form):
    assert (form.kind, form.id, form.name) == ("cv-form", "weekly-tally", "Weekly tally sheet")
    assert list(form.sections) == ["A", "B", "C"] and form.pages == ("AB", "C")
    assert (form.section_prompt, form.page_prompt) == (
        "tally-weekly-tally",
        "tally-weekly-tally-page",
    )


def test_slug():
    assert slug("Screened_6-11 m_Male") == "screened_6_11_m_male"
    assert slug("  Week starting ") == "week_starting"


def test_field_without_type_is_a_count(form):
    assert form.sections["B"].fields[0].type == "count"


def test_prompts_can_be_named():
    data = {**description(), "prompts": {"section": "my-sections", "page": "my-pages"}}
    form = parse_form(data)
    assert form.prompt_spec(False)[0] == "my-sections" and form.prompt_spec(True)[0] == "my-pages"


def change(**sections):
    data = description()
    data["sections"].update(sections)
    return data


@pytest.mark.parametrize(
    "data, message",
    [
        (
            change(D={"fields": [{"key": "X", "type": "number"}]}),
            r"sections.D.fields\[0\] \(X\): type 'number' is not one of",
        ),
        (
            change(D={"fields": [{"key": "Day", "type": "choice"}]}),
            r'sections.D.fields\[0\] \(Day\): a choice needs its "options"',
        ),
        (
            change(D={"fields": [{"key": "region"}]}),
            r"\(region\): another field has the same id 'region'",
        ),
        (change(D={"fields": [{"key": "Region!"}]}), "same id 'region'"),
        (change(DE={"fields": []}), "'DE' is not one letter"),
        (change(D={"title": "x"}), 'sections.D: a section has a list of "fields"'),
        ({**description(), "pages": ["AB", "CD"]}, "section 'D' on page 2 is not in the form"),
        ({**description(), "pages": ["AB", "BC"]}, "section 'B' is on more than one page"),
        ({"id": "x", "sections": {}}, '"sections" is an object'),
    ],
)
def test_form_validation_errors(data, message):
    with pytest.raises(FormError, match=message):
        parse_cv_form(data)


@pytest.mark.parametrize("pages", [["AB", "c"], [["a", "B"], ["C"]], ["Ab", ["C"]]])
def test_pages_as_strings_or_lists(pages):
    assert parse_cv_form({**description(), "pages": pages}).pages == ("AB", "C")


def test_without_pages_the_form_is_one_page():
    data = description()
    data.pop("pages")
    assert parse_cv_form(data).pages == ("ABC",)


def test_section_lowercase_letters_are_upper_cased():
    data = description()
    data["sections"] = {k.lower(): v for k, v in data["sections"].items()}
    data["pages"] = ["ab", "c"]
    assert list(parse_cv_form(data).sections) == ["A", "B", "C"]


def test_section_photo_naming(form):
    items, skipped = form.items(SAMPLES, pages=False)
    assert [i["input"] for i in items] == [
        {"photo": "sheet-01_A.jpg", "section": "A"},
        {"photo": "sheet-01_B.jpg", "section": "B"},
        {"photo": "sheet-01_C.jpg", "section": "C"},
    ]
    assert items[1] == {
        "id": "samples/sheet-01_B.jpg",
        "input": {"photo": "sheet-01_B.jpg", "section": "B"},
        "expected_output": {"screened_6_11_m_male": "12", "screened_6_11_m_female": "0"},
        "metadata": {"sheet": "sheet-01", "section": "B"},
    }
    # whole-page photos of sheet-01 are passed over silently
    assert skipped == [
        "notes_p9.jpg (name it <sheet>_<section>.jpg, sections A, B, C)",
        "sheet-02_A.jpg (no sheet-02.json)",
    ]


def test_page_photo_naming(form):
    items, skipped = form.items(SAMPLES, pages=True)
    assert items == [
        {
            "id": "samples/sheet-01.jpg",
            "input": {"photo": "sheet-01.jpg", "page": 1, "sections": "AB"},
            "expected_output": {
                "week_starting": "02/03/26",
                "region": "North",
                "facility": "EXAMPLE HEALTH CENTRE",
                "facility_code": "0042",
                "screened_6_11_m_male": "12",
                "screened_6_11_m_female": "0",
            },
            "metadata": {"sheet": "sheet-01", "page": 1, "sections": "AB"},
        },
        {
            "id": "samples/sheet-01_p2.jpg",
            "input": {"photo": "sheet-01_p2.jpg", "page": 2, "sections": "C"},
            "expected_output": {"supervisor_signature": "signed"},
            "metadata": {"sheet": "sheet-01", "page": 2, "sections": "C"},
        },
    ]
    rule = "name it <sheet>.jpg for page 1 or <sheet>_p<n>.jpg, n from 1 to 2"
    assert skipped == [f"notes_p9.jpg ({rule})", f"sheet-02_A.jpg ({rule})"]


def test_section_letter_not_in_form_is_a_page_photo(form, tmp_path):
    folder = tmp_path / "photos"
    tiny_image(folder / "sheet-01_Z.jpg")
    tiny_image(folder / "sheet-01-p2.png")
    (folder / "sheet-01_Z.json").write_text("{}")
    (folder / "sheet-01.truth.json").write_text('[{"Region": "South"}]')
    items, skipped = form.items(folder, pages=True)
    assert [(i["input"]["photo"], i["input"]["page"]) for i in items] == [
        ("sheet-01-p2.png", 2),
        ("sheet-01_Z.jpg", 1),
    ]
    assert items[1]["metadata"]["sheet"] == "sheet-01_Z" and skipped == []
    assert form.items(folder, pages=False) == ([], [])


def test_sheet_truth(form, tmp_path):
    good = tmp_path / "s.json"
    good.write_text('[{"region": "North", "Screened_6-11 m_Male": 4}]')
    assert form.load_sheet_truth(good) == {"region": "North", "screened_6_11_m_male": 4}
    bad = tmp_path / "t.json"
    bad.write_text('{"Colour": "red"}')
    with pytest.raises(FormError, match=f"^{bad}: unknown field 'Colour'"):
        form.load_sheet_truth(bad)
    rows = tmp_path / "u.json"
    rows.write_text("[{}, {}]")
    with pytest.raises(FormError, match="one object of field -> value"):
        form.load_sheet_truth(rows)
    assert len(form.check_truths(tmp_path)) == 2


def test_section_schema(form):
    schema = schema_for([form.sections["A"]])
    assert list(schema["properties"]) == ["week_starting", "region", "facility", "facility_code"]
    assert schema["required"] == list(schema["properties"])
    assert schema["additionalProperties"] is False
    assert schema["properties"]["region"] == {
        "type": ["string", "null"],
        "description": "Region: the option whose circle is filled: North, South",
    }


def test_page_schema_holds_every_field_of_the_page(form):
    schema = schema_for(form.page_sections(1))
    assert list(schema["properties"]) == [
        "week_starting",
        "region",
        "facility",
        "facility_code",
        "screened_6_11_m_male",
        "screened_6_11_m_female",
    ]
    assert schema["properties"]["screened_6_11_m_male"]["description"] == (
        "Screened_6-11 m_Male: number written in the boxes"
    )


def test_section_prompt_variables(form):
    assert section_variables(form.sections["B"]) == {
        "section": "B",
        "title": "Screening",
        "fields": (
            "- screened_6_11_m_male: Screened, 6-11 m, Male (number written in the boxes)\n"
            "- screened_6_11_m_female: Screened, 6-11 m, Female (number written in the boxes)"
        ),
    }
    assert section_variables(form.sections["C"])["fields"] == (
        '- supervisor_signature: Supervisor signature ("signed" if signed or stamped, '
        "otherwise null)"
    )


def test_page_prompt_variables(form):
    assert page_variables(form.page_sections(1)) == {
        "fields": (
            'Section A, "Session details":\n'
            "- week_starting: Week starting (date as written)\n"
            "- region: Region (the option whose circle is filled: North, South)\n"
            "- facility: Facility (text as written)\n"
            "- facility_code: Facility code (digits as written)\n"
            "\n"
            'Section B, "Screening":\n'
            "- screened_6_11_m_male: Screened, 6-11 m, Male (number written in the boxes)\n"
            "- screened_6_11_m_female: Screened, 6-11 m, Female (number written in the boxes)"
        )
    }


def test_page_reading_is_split_per_section(form, monkeypatch, tmp_path):
    from formbench import model
    from formbench.forms.base import ReadOptions
    from tests.conftest import make_prompt

    calls = []

    def fake(image, schema, prompt_text, model_tag, **options):
        calls.append({"image": image, "schema": schema, "text": prompt_text, **options})
        return json.dumps(
            {
                "week_starting": "2/3/26",
                "region": " North ",
                "facility": "null",
                "facility_code": None,
                "screened_6_11_m_male": "12",
                "screened_6_11_m_female": "",
            }
        )

    monkeypatch.setattr(model, "complete_json", fake)
    prompt = make_prompt("tally-weekly-tally-page", 4, "Read:\n{{fields}}", ["production"])
    item = {"photo": "sheet-01.jpg", "page": 1, "sections": "AB"}
    result = form.read(tmp_path, item, "small:1b", prompt, ReadOptions(think=True, max_side=900))
    assert result["json_valid"] and result["errors"] == []
    assert result["reading"] == {
        "A": {
            "week_starting": "2/3/26",
            "region": "North",
            "facility": None,
            "facility_code": None,
        },
        "B": {"screened_6_11_m_male": "12", "screened_6_11_m_female": None},
    }
    call = calls[0]
    assert call["image"] == tmp_path / "sheet-01.jpg"
    assert call["text"] == "Read:\n" + page_variables(form.page_sections(1))["fields"]
    assert (call["trace_name"], call["metadata"]) == (
        "tally-weekly-tally-page",
        {"prompt_version": 4, "page": 1},
    )
    assert (call["think"], call["max_side"], call["langfuse_prompt"]) == (True, 900, prompt)
    counts, outcomes = form.score(
        item,
        result["reading"],
        {
            "week_starting": "02/03/26",
            "region": "North",
            "facility": "EXAMPLE HEALTH CENTRE",
            "facility_code": "0042",
            "screened_6_11_m_male": "12",
            "screened_6_11_m_female": "0",
        },
        True,
    )
    assert (counts["field_cells"], counts["field_correct"], counts["field_fn"]) == (5, 3, 2)
    assert [(o["group"], o["field"], o["result"]) for o in outcomes if o["result"] != "right"] == [
        ("A", "facility", "missed"),
        ("A", "facility_code", "missed"),
    ]


def test_section_read_uses_section_prompt(form, monkeypatch, tmp_path):
    from formbench import model
    from formbench.forms.base import ReadOptions
    from tests.conftest import make_prompt

    calls = []
    monkeypatch.setattr(
        model, "complete_json", lambda *a, **kw: calls.append((a, kw)) or "no json here"
    )
    prompt = make_prompt("tally-weekly-tally", 2, "{{section}} {{title}}\n{{fields}}", [])
    result = form.read(tmp_path, {"photo": "x_C.jpg", "section": "C"}, "m", prompt, ReadOptions())
    assert result == {
        "reading": {"C": {"supervisor_signature": None}},
        "json_valid": False,
        "errors": ["no JSON object in the reply"],
        "raw": "no json here",
    }
    args, options = calls[0]
    assert args[2].startswith("C Sign-off\n- supervisor_signature")
    assert options["metadata"] == {"prompt_version": 2, "section": "C"}
    assert options["trace_name"] == "tally-weekly-tally" and options["max_side"] == 4032


def test_score_outcomes_and_empty_sections(form):
    item = {"photo": "sheet-01_A.jpg", "section": "A"}
    truth = {
        "week_starting": "02/03/26",
        "region": "North",
        "facility": None,
        "facility_code": "0042",
    }
    reading = {
        "A": {
            "week_starting": "2-3-2026",
            "region": "South",
            "facility": "X",
            "facility_code": None,
        }
    }
    counts, outcomes = form.score(item, reading, truth, True)
    assert [o["result"] for o in outcomes] == ["right", "misread", "invented", "missed"]
    assert counts == {
        "pages": 1,
        "json_valid": 1.0,
        "field_cells": 4,
        "field_correct": 1,
        "field_tp": 1,
        "field_fp": 2,
        "field_fn": 2,
    }
    blank = {"A": dict.fromkeys(truth)}
    assert form.is_empty_reading(item, blank, truth) == ["A"]
    assert form.is_empty_reading(item, reading, truth) == []
    page = {"photo": "p.jpg", "page": 1, "sections": "AB"}
    assert form.is_empty_reading(
        page, {"A": reading["A"]}, {**truth, "screened_6_11_m_male": "3"}
    ) == ["B"]


def test_check_truths_on_copied_folder(form, tmp_path):
    folder = tmp_path / "samples"
    shutil.copytree(SAMPLES, folder)
    assert form.check_truths(folder) == []
