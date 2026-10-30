import datetime as dt
import json

import pytest

from formbench.evaluation.bench import _values
from formbench.evaluation.review import flag_catch, review_and_approve, review_counts
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
    assert (c["rows"], c["rows_found"], c["rows_invented"]) == (2, 1, 1)
    assert (c["written_cells"], c["written_missed"]) == (2, 1)  # row 2 was not found
    assert c["blank_filled"] == 1  # opv2 made up in row 1


def value(day, count, de="OPV"):
    return {"day": day, "data_element": de, "coc": "c", "data_element_name": de, "value": count}


def test_value_scores():
    v = value
    s = compare_values(_values([v("1", 2), v("2", 1)]), _values([v("1", 2), v("3", 4)]))
    assert s == {"values": 3, "values_exact": 1, "values_missing": 1, "values_extra": 1}


def test_a_perfect_reader_scores_perfectly(scenario, truth, ri_metadata, tmp_path):
    """Feeding the true readings through review must reproduce the expected payload and
    catch every injected problem: this checks the benchmark itself."""
    spec, labels = load_spec(ri_metadata), names(ri_metadata)
    drafts, alarms = [], 0
    for week in truth:
        pages = [
            {"page_id": p["page_id"], **json.loads(json.dumps(p["reading"]))}
            for p in week["photos"]
        ]
        first, approved = review_and_approve(
            pages, week["photos"], COVER, spec, labels, week, go_live=GO_LIVE,
            intended_option="Outreach/Mobile", state_root=tmp_path,
        )  # fmt: skip
        drafts.append(first)
        review = review_counts(first["flags"], pages, week["photos"], edges(scenario))
        assert review["misreads"] == 0
        alarms += review["false_alarms"]
        got = approved["values"] + approved["corrections"]
        want = week["expected"]["values"] + week["expected"]["corrections"]
        scores = compare_values(_values(got), _values(want))
        assert scores["values_exact"] == scores["values"]
    catch = flag_catch(drafts, edges(scenario), cover_flag_expected=True)
    assert catch["edges_caught"] == catch["edges"] and alarms == 0


def test_flags_on_misreads_count_as_caught_and_others_as_false_alarms():
    truth = [{"page_id": "p", "reading": {"rows": [
        {"serial": "1", "sex": "F", "birth_date": "01-06-26", "opv1": "12-07-26"},
        {"serial": "2", "sex": "M", "birth_date": "02-06-26", "opv1": "12-07-26"},
    ]}}]  # fmt: skip
    pages = [{"page_id": "p", "rows": [
        {"serial": "1", "sex": "F", "birth_date": "01-06-26", "opv1": "17-07-26"},  # misread
        {"serial": "2", "sex": "M", "birth_date": "02-06-26", "opv1": "12-07-26"},
    ]}]  # fmt: skip
    flag = {"blocking": True, "page": "p", "code": "date_unreadable", "column": "opv1"}
    counts = review_counts([{**flag, "serial": "1"}, {**flag, "serial": "2"}], pages, truth, [])
    assert counts == {"misreads": 1, "misreads_flagged": 1, "false_alarms": 1}
    # A re-read that disagrees with an approved reading is never a false alarm.
    changed = {**flag, "code": "changed_since_approved", "serial": "2"}
    assert review_counts([changed], pages, truth, [])["false_alarms"] == 0


def test_cli_accepts_every_option_the_commands_use():
    from formbench.cli import build_parser

    args = build_parser().parse_args(
        ["bench", "--scenario", "s", "--engine", "ollama:m", "--max-side", "4096", "--think"]
    )
    assert (args.max_side, args.engine, args.think) == (4096, ["ollama:m"], True)
    assert args.rescore is None and args.out.name == "bench"
    args = build_parser().parse_args(["eval", "--engine", "ollama:m", "--image", "p.jpg"])
    assert args.truth is None and args.metadata.name == "metadata.json"
    args = build_parser().parse_args(["compare", "a.json", "b.json"])
    assert len(args.runs) == 2


def test_engine_must_name_an_ollama_model():
    from formbench.cli import model_tag

    assert model_tag("ollama:hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q8_0") == (
        "hf.co/ibm-granite/granite-vision-4.1-4b-GGUF:Q8_0"
    )
    with pytest.raises(SystemExit):
        model_tag("qwen3.5:4b")


def test_summary_sums_counts_across_scenarios():
    from formbench.evaluation.bench import combine

    def run(cer_edits, chars, pages, values_exact):
        counts = {"cer_edits": cer_edits, "cer_chars": chars, "pages": pages, "values": 10,
                  "values_exact": values_exact, "blocking_flags": 4}  # fmt: skip
        weeks = [{"pages": [{"json_valid": True}] * pages}] * 2
        return {"settings": {"engine": "ollama:m", "scenario": "s"}, "counts": counts, "weeks": weeks, "latency_s": 10}  # fmt: skip

    combined = combine([run(1, 10, 1, 10), run(9, 90, 3, 0)])["ollama:m"]
    assert combined["cer"] == 0.1  # 10 / 100 characters, not the mean of two rates
    assert combined["value_accuracy_after_review"] == 0.5 and combined["flags_per_week"] == 2
    assert combined["latency_s_per_page"] == 5.0
