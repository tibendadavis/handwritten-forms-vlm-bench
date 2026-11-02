import datetime as dt
import json

import pytest

from formbench.evaluation.bench import _values
from formbench.evaluation.review import review_and_approve, review_counts
from formbench.evaluation.scoring import compare_values, same_value, score_page
from formbench.register.spec import load_spec
from formbench.synthetic.simulate import build_truth, simulate
from formbench.weekly.draft import names

GO_LIVE = dt.date(2026, 9, 7)
COVER = {"facility_name": "X", "mfl_code": "MFL-1", "facility_type": "Outrech",
         "district": "", "region": ""}  # fmt: skip


@pytest.fixture(scope="module")
def scenario():
    sim = simulate(3, 3, 3, GO_LIVE)
    return sim


@pytest.fixture
def truth(scenario, ri_metadata):
    return build_truth(scenario, ri_metadata, COVER, ["dash"] * len(scenario["book"]))


def edges(sim):
    return [{"code": e["code"], "serial": str(e["child"].serial)} for e in sim["edges"]]


def test_same_value_normalizes_dates_sex_and_text():
    assert same_value("05-09-26", "5/9/26", "opv1")
    assert not same_value("05-09-26", "06-09-26", "opv1")
    assert same_value("F", "f", "sex") and same_value(None, " ", "opv1")
    assert not same_value(None, "05-09-26", "opv1")


def test_row_and_cell_counts():
    truth = {"cohort_month": "June", "cohort_year": "2026",
             "rows": [{"serial": "1", "sex": "F", "birth_date": "01-06-26", "card_number": "7", "opv1": "12-07-26", "opv2": None},
                      {"serial": "2", "sex": "M", "birth_date": "02-06-26", "card_number": "8", "opv1": "13-07-26"}]}  # fmt: skip
    pred = {"cohort_month": "6", "cohort_year": "26",
            "rows": [{"serial": "1", "sex": "F", "birth_date": "1/6/26", "card_number": "7", "opv1": "12-07-26", "opv2": "19-07-26"},
                     {"serial": "9", "sex": "M"}]}  # fmt: skip
    c = score_page(pred, truth, json_valid=True)
    assert (c["rows"], c["rows_found"]) == (2, 1)
    # Header 2 right; row 1: 5 right + opv2 made up; row 9 paired with row 2: only sex right.
    assert (c["field_correct"], c["field_cells"]) == (8, 13)
    assert (c["field_tp"], c["field_fp"], c["field_fn"]) == (8, 2, 4)


def value(day, count, de="OPV"):
    return {"day": day, "data_element": de, "coc": "c", "data_element_name": de, "value": count}


def test_value_scores():
    v = value
    s = compare_values(_values([v("1", 2), v("2", 1)]), _values([v("1", 2), v("3", 4)]))
    assert s == {"values": 3, "values_exact": 1}


def test_a_perfect_reader_scores_perfectly(scenario, truth, ri_metadata, tmp_path):
    """Feeding the true readings through review must reproduce the expected payload: this
    checks the benchmark itself."""
    spec, labels = load_spec(ri_metadata), names(ri_metadata)
    for week in truth:
        pages = [
            {"page_id": p["page_id"], **json.loads(json.dumps(p["reading"]))}
            for p in week["photos"]
        ]
        first, approved = review_and_approve(
            pages, week["photos"], COVER, spec, labels, week, go_live=GO_LIVE,
            intended_option="Outreach/Mobile", state_root=tmp_path,
        )  # fmt: skip
        review = review_counts(first["flags"], pages, week["photos"])
        assert review["misreads"] == review["misreads_flagged"] == 0
        got = approved["values"] + approved["corrections"]
        want = week["expected"]["values"] + week["expected"]["corrections"]
        scores = compare_values(_values(got), _values(want))
        assert scores["values_exact"] == scores["values"]


def test_a_flag_on_a_misread_counts_as_caught():
    truth = [{"page_id": "p", "reading": {"rows": [
        {"serial": "1", "sex": "F", "birth_date": "01-06-26", "opv1": "12-07-26"},
        {"serial": "2", "sex": "M", "birth_date": "02-06-26", "opv1": "12-07-26"},
    ]}}]  # fmt: skip
    pages = [{"page_id": "p", "rows": [
        {"serial": "1", "sex": "F", "birth_date": "01-06-26", "opv1": "17-07-26"},  # misread
        {"serial": "2", "sex": "M", "birth_date": "02-06-26", "opv1": "12-07-26"},
    ]}]  # fmt: skip
    flag = {"blocking": True, "resolved": False, "page": "p", "code": "date_unreadable", "column": "opv1"}  # fmt: skip
    counts = review_counts([{**flag, "serial": "1"}, {**flag, "serial": "2"}], pages, truth)
    assert counts == {"misreads": 1, "misreads_flagged": 1, "blocking_flags": 2}
    assert review_counts([{**flag, "serial": "2"}], pages, truth)["misreads_flagged"] == 0


def test_cli_accepts_every_option_the_commands_use():
    from formbench.cli import build_parser

    args = build_parser().parse_args(
        [
            "bench",
            "--folder",
            "s",
            "--engine",
            "ollama:m",
            "--max-side",
            "4096",
            "--think",
            "--prompt",
            "3",
        ]
    )
    assert (args.max_side, args.engine, args.think, args.prompt) == (4096, ["ollama:m"], True, "3")
    assert args.rescore is None and args.out.name == "bench"
    assert build_parser().parse_args(["bench", "--scenario", "s"]).folder.name == "s"
    args = build_parser().parse_args(["eval", "--engine", "ollama:m", "--image", "p.jpg"])
    assert args.truth is None and args.metadata.name == "metadata.json" and args.prompt is None
    args = build_parser().parse_args(["compare", "a.json", "b.json"])
    assert len(args.runs) == 2
    args = build_parser().parse_args(["compare", "--folder", "my-photos"])
    assert args.runs == [] and args.folder == "my-photos"


def test_engine_must_name_an_ollama_model():
    from formbench.cli import model_tag

    assert model_tag("ollama:hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q8_0") == (
        "hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q8_0"
    )
    with pytest.raises(SystemExit):
        model_tag("qwen3.5:4b")


def test_summary_sums_counts_across_scenarios():
    from formbench.evaluation.bench import combine

    def run(correct, cells, pages, values_exact):
        counts = {"field_correct": correct, "field_cells": cells, "pages": pages, "values": 10,
                  "values_exact": values_exact, "blocking_flags": 4}  # fmt: skip
        weeks = [{"pages": [{"json_valid": True}] * pages}] * 2
        return {"settings": {"engine": "ollama:m", "scenario": "s"}, "counts": counts, "weeks": weeks, "latency_s": 10}  # fmt: skip

    combined = combine([run(1, 10, 1, 10), run(9, 90, 3, 0)])["ollama:m"]
    assert combined["overall_accuracy"] == 0.1  # 10 / 100 cells, not the mean of two rates
    assert combined["reporting_accuracy_after_review"] == 0.5 and combined["review_workload"] == 2
    assert combined["processing_time"] == 5.0


def test_each_prompt_version_of_a_model_is_its_own_column():
    from formbench.evaluation.bench import combine

    def run(version, correct):
        settings = {"engine": "ollama:m", "scenario": "s", "prompt_version": version}
        return {"settings": settings, "counts": {"field_correct": correct, "field_cells": 10, "pages": 1}}  # fmt: skip

    combined = combine([run(1, 5), run(2, 9), run(2, 7), run(None, 1)])
    assert set(combined) == {"ollama:m prompt v1", "ollama:m prompt v2", "ollama:m"}
    assert combined["ollama:m prompt v2"]["overall_accuracy"] == 0.8


class FakeLangfuse:
    """Runs an experiment in-process the way Langfuse does: task, then evaluators, per item."""

    def __init__(self):
        self.run_scores, self.datasets = [], {}

    def get_dataset(self, name):
        from types import SimpleNamespace

        if name not in self.datasets:
            raise LookupError(name)
        return SimpleNamespace(items=list(self.datasets[name].values()))

    def create_dataset(self, *, name, **_):
        self.datasets[name] = {}

    def create_dataset_item(self, *, dataset_name, id, input, expected_output, **_):
        from types import SimpleNamespace

        item = SimpleNamespace(id=id, input=input, expected_output=expected_output)
        self.datasets[dataset_name][id] = item  # same id: updated, not duplicated

    def update_current_trace(self, **_):
        pass

    def run_experiment(self, *, data, task, evaluators, **_):
        from types import SimpleNamespace

        results = []
        for item in data:
            output = task(item=item)
            local = isinstance(item, dict)  # local items are dicts, dataset items objects
            expected = item["expected_output"] if local else item.expected_output
            for evaluator in evaluators:
                evaluator(input=item["input"] if local else item.input, output=output, expected_output=expected)  # fmt: skip
            results.append(SimpleNamespace(item=item, output=output))
        return SimpleNamespace(item_results=results, dataset_run_id="run-1", dataset_run_url="url")

    def create_score(self, **score):
        self.run_scores.append(score)

    def flush(self):
        pass


def perfect_reader(monkeypatch, answers):
    import formbench.evaluation.bench as bench

    def read_item(folder, item, model, **_):
        if item["kind"] == "cover":
            return {"cover": answers[item["photo"]], "json_valid": True, "errors": [], "latency_s": 1.0}  # fmt: skip
        reading = json.loads(json.dumps(answers[item["photo"]]))
        return {"reading": reading, "json_valid": True, "errors": [], "latency_s": 2.0}

    monkeypatch.setattr(bench, "read_item", read_item)


def test_a_scenario_run_scores_a_perfect_reader_perfectly(scenario, truth, ri_metadata, tmp_path, monkeypatch):  # fmt: skip
    from formbench.evaluation.bench import load_items, run_bench

    folder = tmp_path / "scenario"
    folder.mkdir()
    for n, week in enumerate(truth, 1):
        for photo in week["photos"]:
            photo["file"] = f"week-{n:02d}/page-{photo['page_id']}.jpg"
    full = {"go_live": GO_LIVE.isoformat(), "cover": COVER, "intended_option": "Outreach/Mobile",
            "edges": edges(scenario), "weeks": truth}  # fmt: skip
    (folder / "truth.json").write_text(json.dumps(full))
    items, _, _ = load_items(folder)
    perfect_reader(monkeypatch, {i["input"]["photo"]: i["expected_output"] for i in items})
    langfuse = FakeLangfuse()
    run_bench(folder, "m", ri_metadata, langfuse=langfuse, configs={})
    result = run_bench(folder, "m", ri_metadata, langfuse=langfuse, configs={})
    assert len(langfuse.datasets["scenario"]) == len(items)  # re-syncing does not duplicate
    m = result["metrics"]
    assert m["reporting_accuracy_after_review"] == 1 and m["reporting_accuracy"] == 1
    assert m["overall_accuracy"] == 1 and "error_detection_rate" not in m  # nothing misread to flag
    assert m["processing_time"] == 2.0 and result["page_count"] == len(items) - 1
    assert {s["name"] for s in langfuse.run_scores} == set(m)
    assert len(langfuse.run_scores) == 2 * len(m)
    assert all(s["dataset_run_id"] == "run-1" for s in langfuse.run_scores)


def test_a_photo_folder_needs_a_truth_file_per_photo(ri_metadata, tmp_path, monkeypatch):
    from formbench.evaluation.bench import load_items, run_bench

    reading = {"cohort_month": "June", "cohort_year": "2026",
               "rows": [{"serial": "1", "sex": "F", "birth_date": "01-06-26", "opv1": "12-07-26"}]}  # fmt: skip
    for name in ("a.jpg", "b.JPEG", "no-truth.png"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "a.json").write_text(json.dumps(reading))
    (tmp_path / "b.truth.json").write_text(json.dumps(reading))
    items, truth, skipped = load_items(tmp_path)
    assert [i["input"]["photo"] for i in items] == ["a.jpg", "b.JPEG"] and truth is None
    assert skipped == ["no-truth.png"]
    perfect_reader(monkeypatch, {"a.jpg": reading, "b.JPEG": reading})
    result = run_bench(tmp_path, "m", ri_metadata, langfuse=FakeLangfuse(), configs=None)
    m = result["metrics"]
    assert (
        m["overall_accuracy"] == 1 and m["rows_detected"] == 1 and m["processing_success_rate"] == 1
    )
    assert "reporting_accuracy_after_review" not in m and "review_workload" not in m
    assert result["skipped"] == ["no-truth.png"] and result["failed"] == []


def test_score_definitions_are_created_once():
    from types import SimpleNamespace

    from formbench.evaluation.scoring import SCORES
    from formbench.evaluation.tracking import score_configs

    created = []

    class ScoreConfigs:
        def get(self, page, limit):
            data = [SimpleNamespace(name="overall_accuracy", id="c1", is_archived=False)]
            return SimpleNamespace(data=data, meta=SimpleNamespace(total_pages=1))

        def create(self, request):
            created.append(request)
            return SimpleNamespace(id=f"id-{request.name}")

    configs = score_configs(SimpleNamespace(api=SimpleNamespace(score_configs=ScoreConfigs())))
    assert configs["overall_accuracy"] == "c1" and len(configs) == len(SCORES)
    assert "overall_accuracy" not in {r.name for r in created} and len(created) == len(SCORES) - 1
    rows = next(r for r in created if r.name == "rows_detected")
    assert rows.max_value == 1 and rows.description
