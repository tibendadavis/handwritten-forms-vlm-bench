import json

import pytest

from formbench.tally.form import (
    DEFAULT_PROMPT,
    compile_prompt,
    load_sheet_truth,
    parse_form,
    prompt_variables,
    same,
    score_section,
    section_schema,
)

FORM = {
    "id": "test-tally",
    "name": "Test tally sheet",
    "sections": {
        "A": {"title": "Session", "fields": [
            {"key": "Week starting", "type": "date"},
            {"key": "Site code", "type": "digits"},
            {"key": "Site name", "type": "text"},
            {"key": "Area", "type": "choice", "options": ["North", "S.East"]},
        ]},
        "B": {"title": "Screening", "fields": [
            {"key": "Screened_<6 m_Male", "type": "count"},
            {"key": "Screened_<6 m_Female", "type": "count"},
            {"key": "Officer signature", "type": "mark"},
        ]},
    },
}  # fmt: skip
SHEET = {"Week starting": "19/10/2026", "Site code": "004217", "Site name": "CENTRAL CLINIC",
         "Area": "S.East", "Screened_<6 m_Male": 7, "Screened_<6 m_Female": "", "Officer signature": "signed"}  # fmt: skip


@pytest.fixture
def form():
    return parse_form(FORM)


def test_each_section_asks_only_for_its_own_fields(form):
    schema = section_schema(form.sections["B"])
    assert list(schema["properties"]) == ["screened_6_m_male", "screened_6_m_female", "officer_signature"]  # fmt: skip
    assert schema["additionalProperties"] is False and schema["required"] == list(schema["properties"])  # fmt: skip
    prompt = compile_prompt(DEFAULT_PROMPT, prompt_variables(form.sections["A"]))
    assert 'section A, "Session"' in prompt and "- area: Area (the option whose circle is filled: North, S.East)" in prompt  # fmt: skip
    assert "{{" not in prompt and "screened" not in prompt


def test_mistakes_in_the_form_description_are_explained():
    bad = json.loads(json.dumps(FORM))
    bad["sections"]["B"]["fields"].append({"key": "Screened <6 m Male", "type": "count"})
    with pytest.raises(ValueError, match="same name"):
        parse_form(bad)
    with pytest.raises(ValueError, match="not one of"):
        parse_form({"id": "x", "sections": {"A": {"fields": [{"key": "a", "type": "tick"}]}}})
    with pytest.raises(ValueError, match="needs its options"):
        parse_form({"id": "x", "sections": {"A": {"fields": [{"key": "a", "type": "choice"}]}}})


def test_a_sheet_truth_is_one_object_named_as_in_the_form(form, tmp_path):
    path = tmp_path / "sheet-01.json"
    path.write_text(json.dumps([SHEET]))  # a list of one sheet works too
    truth = load_sheet_truth(path, form)
    assert truth["screened_6_m_male"] == 7 and truth["week_starting"] == "19/10/2026"
    path.write_text(json.dumps({**SHEET, "Screened girls": 3}))
    with pytest.raises(SystemExit, match="unknown field 'Screened girls'"):
        load_sheet_truth(path, form)
    path.write_text('{"Site code": 004217}')
    with pytest.raises(SystemExit, match="need quotes"):
        load_sheet_truth(path, form)


def test_values_are_compared_by_what_they_mean(form):
    f = {fd.key: fd for s in form.sections.values() for fd in s.fields}
    assert same(f["Screened_<6 m_Male"], 7, "007") and same(f["Screened_<6 m_Male"], "", "0")
    assert not same(f["Screened_<6 m_Male"], 7, "1")
    assert same(f["Site code"], "004217", "0042 17") and not same(f["Site code"], "004217", "4217")
    assert same(f["Week starting"], "19/10/2026", "19-10-26")
    assert same(f["Area"], "S.East", "s east") and same(
        f["Site name"], "CENTRAL CLINIC", "Central  clinic"
    )
    assert same(f["Officer signature"], "signed", "yes") and not same(
        f["Officer signature"], "signed", None
    )


def test_section_scores_use_the_register_counts(form, tmp_path):
    from formbench.evaluation.scoring import metrics

    path = tmp_path / "sheet-01.json"
    path.write_text(json.dumps(SHEET))
    truth = load_sheet_truth(path, form)
    read = {"screened_6_m_male": "7", "screened_6_m_female": "2", "officer_signature": None}
    m = metrics(score_section(read, truth, form.sections["B"], True))
    # cells: male right, female invented, signature missed
    assert m["overall_accuracy"] == round(1 / 3, 4) and m["completeness"] == 0.5 and m["reliability"] == 0.5  # fmt: skip
    assert "rows_detected" not in m


def test_a_folder_run_scores_each_photo_on_its_own_section(form, tmp_path, monkeypatch):
    from test_evaluation_bench import FakeLangfuse

    import formbench.tally.bench as bench

    folder = tmp_path / "samples"
    folder.mkdir()
    (folder / "sheet-01.json").write_text(json.dumps(SHEET))
    for name in ("sheet-01_A.jpg", "sheet-01_B.jpg", "sheet-01_Z.jpg", "other_A.jpg"):
        (folder / name).touch()
    items, skipped = bench.load_items(folder, form)
    assert [i["input"] for i in items] == [{"photo": "sheet-01_A.jpg", "section": "A"},
                                           {"photo": "sheet-01_B.jpg", "section": "B"}]  # fmt: skip
    assert set(items[1]["expected_output"]) == {"screened_6_m_male", "screened_6_m_female", "officer_signature"}  # fmt: skip
    assert len(skipped) == 2 and any("no other.json" in s for s in skipped)

    def perfect(photo, form, letter, model, **_):
        expected = next(i["expected_output"] for i in items if i["input"]["photo"] == photo.name)
        return {"reading": {k: None if v in (None, "") else str(v) for k, v in expected.items()},
                "json_valid": True, "errors": [], "raw": "{}"}  # fmt: skip

    monkeypatch.setattr(bench, "read_section", perfect)
    langfuse = FakeLangfuse()
    result = bench.run_tally(folder, form, "m", langfuse=langfuse, configs={})
    assert result["metrics"]["overall_accuracy"] == 1 and result["page_count"] == 2
    assert {s["name"] for s in langfuse.run_scores} == set(result["metrics"])
    assert all(s["comment"] for s in langfuse.run_scores)


def test_the_model_gets_the_section_prompt_and_the_trace_its_version(form, monkeypatch):
    from types import SimpleNamespace

    import formbench.vlm as vlm
    from formbench.tally.form import read_section

    calls = []
    monkeypatch.setattr(vlm, "complete_json", lambda *a, **k: calls.append((a, k)) or '{"area": "null"}')  # fmt: skip
    prompt = SimpleNamespace(version=4, compile=lambda **v: f"Section {v['section']}: {v['title']}")
    result = read_section(None, form, "A", "m", max_side=100, prompt=prompt)
    ((args, kwargs),) = calls
    assert args[2] == "Section A: Session" and kwargs["langfuse_prompt"] is prompt
    assert kwargs["name"] == "tally-test-tally" and kwargs["metadata"] == {"prompt_version": 4, "section": "A"}  # fmt: skip
    assert result["reading"]["area"] is None  # the text "null" is an empty field
