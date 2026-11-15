import json

import pytest

from formbench import model
from formbench.forms import kind_of, load_form, parse_form
from formbench.forms.base import FormError, ReadOptions
from formbench.forms.cv_form import Field
from formbench.forms.summary import fields_variable, parse_summary, same_value
from formbench.runner import mode_of
from tests.conftest import FIXTURES, make_prompt, tiny_image

FORM_JSON = FIXTURES / "summary" / "form.json"


def small():
    """A summary form small enough to write out in full."""
    return {
        "kind": "summary",
        "id": "mini",
        "fields": [
            {"key": "District", "type": "text"},
            {"key": "Month"},
            {"key": "BCG_0-11 m_Facility"},
            {"key": "BCG_0-11 m_Outreach"},
            {"key": "Sessions planned", "type": "count"},
        ],
        "personal_words": ["officer"],
    }


@pytest.fixture
def form():
    return parse_summary(small())


def test_full_form_layout(tmp_path):
    (tmp_path / "form.json").write_text(FORM_JSON.read_text())
    (tmp_path / "photos").mkdir()
    form = load_form(tmp_path / "photos")
    assert (form.kind, form.id, form.prompt_name) == ("summary", "mf-06", "summary-page")
    assert len(form.fields) == 165
    ids = {f.id for f in form.fields}
    assert {
        "bcg_0_11_m_facility",
        "sessions_planned_facility",
        "mmn_powder_12_59_m_outreach",
        "child_identified_with_sam_oedema_12_59_m",
    } <= ids
    assert "reporting_officer" not in ids  # personal: never asked for


def test_dates_compare_year_first():
    date = Field("Date Form completed", "date")
    assert same_value(date, "2026-10-01", "2026/10/01")
    assert same_value(date, "2026-10-01", "1/10/26")
    assert not same_value(date, "2026-10-01", "2026/10/03")


def test_kind_from_shape():
    data = small()
    assert parse_form(data).kind == "summary"
    del data["kind"]
    assert kind_of(data) == "summary"


def test_prompt_lines_and_schema(form):
    assert form.prompt_name == "summary-mini"
    assert fields_variable(form.fields) == (
        "- district: District (text as written)\n"
        "- month: Month (number written in the cell)\n"
        "- bcg_0_11_m_facility: BCG, 0-11 m, Facility (number written in the cell)\n"
        "- bcg_0_11_m_outreach: BCG, 0-11 m, Outreach (number written in the cell)\n"
        "- sessions_planned: Sessions planned (number written in the cell)"
    )
    schema = form.schema()
    assert schema["required"] == [f.id for f in form.fields] == list(schema["properties"])
    assert schema["additionalProperties"] is False
    assert schema["properties"]["bcg_0_11_m_facility"] == {
        "type": ["string", "null"],
        "description": "BCG_0-11 m_Facility: number written in the cell",
    }


@pytest.mark.parametrize(
    "data, message",
    [
        ({"kind": "summary", "fields": [{"key": "X"}]}, "'id' is missing"),
        ({"id": "x", "fields": []}, '"fields" is a list'),
        ({"id": "x", "fields": [{"key": "X", "type": "choice"}]}, r"\(X\): type 'choice'"),
        ({"id": "x", "fields": [{"key": "A b"}, {"key": "a-B"}]}, "same id 'a_b'"),
        ({"id": "x", "fields": [{"type": "text"}]}, r"fields\[0\]: a field is an object"),
        ({"id": "x", "fields": [{"key": "--"}]}, "needs letters or digits"),
    ],
)
def test_form_mistakes(data, message):
    with pytest.raises(FormError, match=message):
        parse_summary(data)


def test_truth(form, tmp_path):
    good = tmp_path / "page-01.json"
    good.write_text(
        '{"District": "Example", "BCG_0-11 m_Facility": "49", "Reporting officer": "X"}'
    )
    assert form.load_truth(good) == {
        "district": "Example",
        "month": None,
        "bcg_0_11_m_facility": "49",
        "bcg_0_11_m_outreach": None,
        "sessions_planned": None,
    }
    bad = tmp_path / "page-02.json"
    bad.write_text('{"BCG_12-59 m_Facility": "3"}')
    with pytest.raises(FormError, match=f"^{bad}: unknown field 'BCG_12-59 m_Facility'"):
        form.load_truth(bad)
    assert len(form.check_truths(tmp_path)) == 1


def test_items(form, tmp_path):
    tiny_image(tmp_path / "page-01.jpg")
    tiny_image(tmp_path / "page-02.jpg")
    (tmp_path / "page-01.json").write_text('{"Month": "09"}')
    items, skipped = form.items(tmp_path)
    assert [(i["id"], i["input"]) for i in items] == [
        (f"{tmp_path.name}/page-01.jpg", {"photo": "page-01.jpg", "kind": "page"})
    ]
    assert items[0]["expected_output"]["month"] == "09"
    assert skipped == ["page-02.jpg (no truth file)"]
    with pytest.raises(FormError, match="a summary is always one photo per page"):
        form.items(tmp_path, pages=True)
    assert mode_of(form, False) == "summary"


def test_read_score_and_empty_page(form, monkeypatch, tmp_path):
    calls = []

    def fake(image, schema, prompt_text, model_tag, **options):
        calls.append({"text": prompt_text, **options})
        return json.dumps(
            {
                "district": " Example ",
                "month": "9",
                "bcg_0_11_m_facility": "48",
                "bcg_0_11_m_outreach": "null",
                "sessions_planned": None,
            }
        )

    monkeypatch.setattr(model, "complete_json", fake)
    prompt = make_prompt("summary-mini", 3, "Read:\n{{fields}}", ["production"])
    item = {"photo": "page-01.jpg", "kind": "page"}
    result = form.read(tmp_path, item, "small:1b", prompt, ReadOptions())
    assert result["json_valid"]
    assert result["reading"] == {
        "district": "Example",
        "month": "9",
        "bcg_0_11_m_facility": "48",
        "bcg_0_11_m_outreach": None,
        "sessions_planned": None,
    }
    assert calls[0]["text"] == "Read:\n" + fields_variable(form.fields)
    assert (calls[0]["trace_name"], calls[0]["metadata"]) == ("summary-mini", {"prompt_version": 3})
    truth = {
        "district": "EXAMPLE",
        "month": "09",
        "bcg_0_11_m_facility": "49",
        "bcg_0_11_m_outreach": "0",  # a written 0 and an empty cell are the same
        "sessions_planned": "11",
    }
    counts, outcomes = form.score(item, result["reading"], truth, True)
    assert [(o["group"], o["field"], o["result"]) for o in outcomes] == [
        ("District", "district", "right"),
        ("Month", "month", "right"),
        ("BCG", "bcg_0_11_m_facility", "misread"),
        ("Sessions planned", "sessions_planned", "missed"),
    ]
    assert (counts["field_cells"], counts["field_correct"]) == (4, 2)
    assert form.is_empty_reading(item, dict.fromkeys(truth), truth) == ["page"]
    assert form.is_empty_reading(item, result["reading"], truth) == []
