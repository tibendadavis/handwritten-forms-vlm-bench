import json
import re
import shutil
from pathlib import Path

import httpx
import openai
import pytest

from formbench import model
from formbench.forms import load_form
from formbench.forms.base import ReadOptions
from formbench.runner import model_slug, prepare, rescore, run_label, run_model
from formbench.scoring import SCORES
from formbench.tracking import RunError
from tests.conftest import FIXTURES, FakeLangfuse

ROOT = Path(__file__).parent.parent
RESULT_KEYS = [
    "settings",
    "run_name",
    "dataset_run_url",
    "metrics",
    "counts",
    "latency_s",
    "photo_count",
    "photos",
    "outcomes",
    "failed",
    "skipped",
    "empty_readings",
]
PAGE_01 = {
    "cohort_month": "6",
    "cohort_year": "26",
    "rows": [
        {
            "serial": "1",
            "card_number": "0101",
            "sex": "F",
            "birth_date": "1/6/26",
            "age": "2/52",
            "bcg": "02-06-26",
            "opv0": "02-06-26",
        },
        {
            "serial": "2",
            "card_number": "0102",
            "sex": "M",
            "birth_date": "03-06-26",
            "age": "1/52",
            "bcg": "04-06-26",
        },
    ],
}


def register_reply(rows_and_header):
    keys = [
        c["key"] for c in json.loads((FIXTURES / "register" / "form.json").read_text())["columns"]
    ]
    rows = [{k: row.get(k) for k in keys} for row in rows_and_header["rows"]]
    return json.dumps({**rows_and_header, "rows": rows})


@pytest.fixture
def replies(monkeypatch):
    """Scripted model replies by photo name; an exception is raised instead of replying."""
    script, calls = {}, []

    def fake(image, schema, prompt_text, model_tag, **options):
        calls.append(
            {"photo": Path(image).name, "model": model_tag, "text": prompt_text, **options}
        )
        reply = script[Path(image).name]
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(model, "complete_json", fake)
    script["calls"] = calls
    return script


def copy(name, tmp_path):
    shutil.copytree(FIXTURES / name, tmp_path / name)
    return tmp_path / name


def go(langfuse, folder, form, models, *, pages=False, choice=None, options=None, out=None):
    options = options or ReadOptions()
    plan = prepare(langfuse, folder, form, pages=pages, options=options, prompt_choice=choice)
    return [run_model(plan, m, langfuse, out or folder.parent / "results") for m in models]


def test_full_register_run(tmp_path, replies):
    folder = copy("register", tmp_path) / "pages"
    form = load_form(folder, FIXTURES / "register" / "form.json")
    replies["page-01.jpg"] = register_reply(PAGE_01)
    replies["page-02.jpg"] = register_reply(
        {
            "cohort_month": None,
            "cohort_year": None,
            "rows": [
                {
                    "serial": "3",
                    "sex": "F",
                    "birth_date": "05-06-26",
                    "age": "3/12",
                    "bcg": "06-06-26",
                }
            ],
        }
    )
    langfuse = FakeLangfuse(configs=["overall_accuracy", "completeness", "reliability"])
    (record,) = go(langfuse, folder, form, ["small:1b"])

    assert record["settings"] == {
        "folder": "pages",
        "form": "register",
        "kind": "register",
        "mode": "register",
        "model": "small:1b",
        "think": False,
        "max_side": 4032,
        "prompt_name": "register-page",
        "prompt_version": 1,
    }
    assert re.fullmatch(r"small:1b prompt v1 \d{8}T\d{6}Z", record["run_name"])
    assert record["counts"] == {
        "pages": 2,
        "json_valid": 2.0,
        "field_cells": 22,
        "field_correct": 20,
        "field_tp": 20,
        "field_fp": 0,
        "field_fn": 2,
        "rows": 3,
        "rows_found": 3,
    }
    assert record["metrics"]["overall_accuracy"] == round(20 / 22, 4)
    assert record["metrics"]["rows_detected"] == 1.0
    assert record["skipped"] == ["page-03.jpg (no truth file)"]
    assert record["failed"] == [] and record["empty_readings"] == []
    assert [p["part"] for p in record["photos"]] == [None, None]
    missed = [o for o in record["outcomes"] if o["result"] != "right"]
    assert [(o["field"], o["result"]) for o in missed] == [
        ("card_number", "missed"),
        ("opv0", "missed"),
    ]
    assert missed[1] == {
        "file": "page-02.jpg",
        "group": "opv0",
        "field": "opv0",
        "type": "date",
        "truth": "06-06-26",
        "read": None,
        "result": "missed",
    }

    saved = json.loads(Path(record["path"]).read_text())
    assert list(saved) == RESULT_KEYS
    assert Path(record["path"]).name.endswith("_pages_small_1b.json")
    assert set(saved["photos"][0]) == {
        "file",
        "part",
        "json_valid",
        "errors",
        "latency_s",
        "reading",
    }

    # Langfuse: configs completed, dataset items, prompt v1, one experiment, run scores
    assert {c.name for c in langfuse.configs} == {s.name for s in SCORES}
    items = langfuse.datasets["pages"]
    assert set(items) == {"pages/page-01.jpg", "pages/page-02.jpg"}
    item = items["pages/page-02.jpg"]
    assert item.input == {"photo": "page-02.jpg", "kind": "page"}
    assert item.metadata == {"kind": "page"}
    assert "redacted" not in json.dumps(item.expected_output)
    assert set(item.expected_output["rows"][0]) == set(form.keys)
    assert langfuse.experiments == [
        {"name": "pages", "run_name": record["run_name"], "max_concurrency": 1}
    ]
    assert langfuse.traces[0] == {
        "name": "small:1b page-01.jpg",
        "tags": ["formbench", "register"],
        "metadata": record["settings"],
    }
    run_scores = {s["name"]: s for s in langfuse.scores}
    assert set(run_scores) == set(record["metrics"])
    accuracy = run_scores["overall_accuracy"]
    assert accuracy["dataset_run_id"] == "run-1" and accuracy["config_id"] == "cfg-overall_accuracy"
    assert accuracy["comment"].endswith("= 20 ÷ 22")
    assert langfuse.flushed == 1
    assert record["dataset_run_url"] == "https://langfuse.test/runs/run-1"
    call = replies["calls"][0]
    assert call["trace_name"] == "register-page" and call["metadata"] == {"prompt_version": 1}
    assert call["langfuse_prompt"].version == 1


def cv_form_reply(**values):
    return json.dumps(values)


def test_full_cv_form_section_run(tmp_path, replies):
    folder = copy("cv-form", tmp_path) / "samples"
    form = load_form(folder)
    replies["sheet-01_A.jpg"] = cv_form_reply(
        week_starting="2/3/26", region="North", facility="EXAMPLE HEALTH CENTRE", facility_code="42"
    )
    replies["sheet-01_B.jpg"] = cv_form_reply(screened_6_11_m_male="12", screened_6_11_m_female="0")
    replies["sheet-01_C.jpg"] = cv_form_reply(supervisor_signature=None)
    langfuse = FakeLangfuse()
    langfuse.add_prompt("tally-weekly-tally", "Section {{section}}\n{{fields}}", ["production"])
    langfuse.add_prompt("tally-weekly-tally", "v2 {{section}} {{title}}\n{{fields}}", ["test"])
    options = ReadOptions(think=True, max_side=2000)
    (record,) = go(langfuse, folder, form, ["small:1b"], choice="2", options=options)

    assert record["settings"]["mode"] == "sections" and record["settings"]["prompt_version"] == 2
    assert re.fullmatch(r"small:1b prompt v2 think max-side 2000 \d{8}T\d{6}Z", record["run_name"])
    assert record["counts"]["field_cells"] == 6 and record["counts"]["field_correct"] == 4
    assert record["metrics"]["reliability"] == 0.8  # a misread is a false positive too
    assert [p["part"] for p in record["photos"]] == ["A", "B", "C"]
    assert [
        (o["group"], o["field"], o["result"]) for o in record["outcomes"] if o["result"] != "right"
    ] == [("A", "facility_code", "misread"), ("C", "supervisor_signature", "missed")]
    assert record["empty_readings"] == [{"file": "sheet-01_C.jpg", "parts": ["C"]}]
    item = langfuse.datasets["samples"]["samples/sheet-01_A.jpg"]
    assert item.input == {"photo": "sheet-01_A.jpg", "section": "A"}
    assert item.metadata == {"sheet": "sheet-01", "section": "A"}
    assert item.expected_output["facility_code"] == "0042"
    calls = replies["calls"]
    assert calls[1]["text"].startswith("v2 B Screening\n- screened_6_11_m_male")
    assert calls[1]["metadata"] == {"prompt_version": 2, "section": "B"}
    assert (calls[1]["think"], calls[1]["max_side"]) == (True, 2000)
    assert record["skipped"] == [
        "notes_p9.jpg (name it <sheet>_<section>.jpg, sections A, B, C)",
        "sheet-02_A.jpg (no sheet-02.json)",
    ]


def test_full_cv_form_page_run_with_two_models(tmp_path, replies):
    folder = copy("cv-form", tmp_path) / "samples"
    form = load_form(folder)
    replies["sheet-01.jpg"] = cv_form_reply(
        week_starting="02/03/26",
        region="North",
        facility="EXAMPLE HEALTH CENTRE",
        facility_code="0042",
        screened_6_11_m_male="12",
        screened_6_11_m_female=None,
    )
    replies["sheet-01_p2.jpg"] = cv_form_reply(supervisor_signature="stamped")
    langfuse = FakeLangfuse()
    records = go(langfuse, folder, form, ["small:1b", "big:7b"], pages=True)
    for record in records:
        assert record["settings"]["mode"] == "pages"
        assert record["counts"]["pages"] == 2 and record["counts"]["field_correct"] == 6
        assert record["metrics"]["overall_accuracy"] == 1.0
        assert [p["part"] for p in record["photos"]] == [1, 2]
    assert re.fullmatch(r"big:7b full pages prompt v1 \d{8}T\d{6}Z", records[1]["run_name"])
    created = [kw["name"] for name, kw in langfuse.calls if name == "create_prompt"]
    assert created == ["tally-weekly-tally-page"]
    item = langfuse.datasets["samples"]["samples/sheet-01_p2.jpg"]
    assert item.input == {"photo": "sheet-01_p2.jpg", "page": 2, "sections": "C"}
    assert item.metadata == {"sheet": "sheet-01", "page": 2, "sections": "C"}
    assert replies["calls"][0]["metadata"] == {"prompt_version": 1, "page": 1}
    assert replies["calls"][0]["trace_name"] == "tally-weekly-tally-page"
    assert len(langfuse.experiments) == 2 and len(langfuse.scores) == 2 * len(records[0]["metrics"])


def connection_error():
    return openai.APIConnectionError(request=httpx.Request("POST", "http://localhost"))


def test_failed_photos_are_recorded_and_scored_as_not_read(tmp_path, replies):
    folder = copy("cv-form", tmp_path) / "samples"
    form = load_form(folder)
    replies["sheet-01_A.jpg"] = connection_error()
    replies["sheet-01_B.jpg"] = cv_form_reply(screened_6_11_m_male="12", screened_6_11_m_female="")
    replies["sheet-01_C.jpg"] = openai.APITimeoutError(request=httpx.Request("POST", "http://x"))
    (record,) = go(FakeLangfuse(), folder, form, ["m"])
    assert record["failed"] == ["sheet-01_A.jpg"]
    assert record["photos"][0]["errors"] == ["not read: the model call failed"]
    assert record["photos"][2]["errors"] == ["model call timed out (no reply)"]
    assert record["counts"]["pages"] == 3 and record["counts"]["json_valid"] == 1.0
    assert record["counts"]["field_fn"] == 5  # four in A and one in C, all missed
    assert record["empty_readings"] == []  # only usable replies are checked


def test_every_photo_failing_stops(tmp_path, replies):
    folder = copy("cv-form", tmp_path) / "samples"
    for name in ("sheet-01_A.jpg", "sheet-01_B.jpg", "sheet-01_C.jpg"):
        replies[name] = connection_error()
    with pytest.raises(RunError, match=r"every photo failed. Is Ollama running \(ollama list\)"):
        go(FakeLangfuse(), folder, load_form(folder), ["m"])


def test_langfuse_unreachable_stops_before_any_model_call(tmp_path, replies):
    folder = copy("cv-form", tmp_path) / "samples"
    with pytest.raises(RunError) as error:
        go(FakeLangfuse(down=True), folder, load_form(folder), ["m"])
    assert str(error.value).startswith(
        "Langfuse is required for runs and is not reachable: connection refused."
    )
    assert replies["calls"] == []


def test_pages_with_a_register_stops(tmp_path):
    folder = copy("register", tmp_path) / "pages"
    form = load_form(folder, FIXTURES / "register" / "form.json")
    with pytest.raises(RunError, match="--pages is for cv-forms"):
        go(FakeLangfuse(), folder, form, ["m"], pages=True)


def test_rescore_reproduces_metrics(tmp_path, replies):
    folder = copy("cv-form", tmp_path) / "samples"
    form = load_form(folder)
    replies["sheet-01_A.jpg"] = cv_form_reply(region="South")
    replies["sheet-01_B.jpg"] = cv_form_reply(screened_6_11_m_male="12")
    replies["sheet-01_C.jpg"] = cv_form_reply(supervisor_signature="signed")
    (record,) = go(FakeLangfuse(), folder, form, ["m"])
    path = Path(record["path"])
    saved = json.loads(path.read_text())
    saved["metrics"], saved["counts"], saved["outcomes"] = {}, {}, []
    path.write_text(json.dumps(saved))
    again = rescore(path, folder, form)
    assert again["metrics"] == record["metrics"] and again["counts"] == record["counts"]
    assert again["outcomes"] == record["outcomes"]
    assert json.loads(path.read_text())["metrics"] == record["metrics"]


def test_labels_and_slugs():
    settings = {
        "model": "qwen3.5:9b",
        "mode": "pages",
        "prompt_version": 5,
        "think": True,
        "max_side": 2048,
    }
    assert run_label(settings) == "qwen3.5:9b full pages prompt v5 think 2048px"
    assert run_label({**settings, "mode": "sections", "think": False, "max_side": 4032}) == (
        "qwen3.5:9b prompt v5"
    )
    assert model_slug("qwen3.5:9b") == "qwen3_5_9b"
