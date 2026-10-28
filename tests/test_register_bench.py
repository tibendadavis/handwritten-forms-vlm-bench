import datetime as dt
import json

import pytest

from formbench.register_bench import (
    flag_catch,
    review_and_approve,
    review_counts,
    same_value,
    score_transcription,
    score_values,
)
from formbench.register_run import names
from formbench.register_spec import load_spec
from formbench.register_synth import build_truth, simulate

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


def test_transcription_counts():
    truth = {"cohort_month": "June", "cohort_year": "2026",
             "rows": [{"serial": "1", "sex": "F", "birth_date": "01-06-26", "card_number": "7", "opv1": "12-07-26", "opv2": None},
                      {"serial": "2", "sex": "M", "birth_date": "02-06-26", "card_number": "8", "opv1": None}]}  # fmt: skip
    pred = {"cohort_month": "6", "cohort_year": "26",
            "rows": [{"serial": "1", "sex": "F", "birth_date": "1/6/26", "card_number": "7", "opv1": "12-07-26", "opv2": "19-07-26"},
                     {"serial": "9", "sex": "M"}]}  # fmt: skip
    c = score_transcription(pred, truth)
    assert (c["rows"], c["rows_found"], c["rows_invented"]) == (2, 1, 1)
    assert (c["date_cells"], c["date_correct"], c["blank_filled"]) == (1, 1, 1)
    assert c["birth_correct"] == 1 and c["cohort_correct"] == 1


def value(day, count, de="OPV"):
    return {"day": day, "data_element": de, "coc": "c", "data_element_name": de, "value": count}


def test_value_scores():
    v = value
    s = score_values([v("1", 2), v("2", 1)], [v("1", 2), v("3", 4)])
    assert s == {"values": 3, "values_exact": 1, "values_missing": 1, "values_extra": 1,
                 "abs_error": 5, "children": 0, "children_exact": 0}  # fmt: skip


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
        scores = score_values(got, want)
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


def test_cli_accepts_every_option_main_uses():
    from formbench.register_bench import build_parser

    args = build_parser().parse_args(
        [
            "--scenario",
            "s",
            "--engine",
            "ollama:m",
            "--mode",
            "full",
            "--max-side",
            "4096",
            "--think",
        ]
    )
    assert (args.max_side, args.mode, args.engine, args.think) == (
        4096,
        ["full"],
        ["ollama:m"],
        True,
    )
    assert args.rescore is None and args.out.name == "bench"


def test_summary_sums_counts_across_scenarios():
    from formbench.register_bench import combine

    def run(cer_edits, chars, pages):
        counts = {"cer_edits": cer_edits, "cer_chars": chars, "pages": pages, "rows": 1,
                  "rows_found": 1, "rows_invented": 0, "date_cells": 1, "date_correct": 1,
                  "blank_cells": 1, "blank_filled": 0, "written_cells": 1, "written_missed": 0,
                  "birth": 1, "birth_correct": 1, "sex_correct": 1, "card_correct": 1,
                  "cohort_correct": pages, "blocking_flags": 0, "edges": 0, "edges_caught": 0,
                  "misreads": 0, "misreads_flagged": 0, "false_alarms": 0, "values": 1,
                  "values_exact": 1, "values_missing": 0, "values_extra": 0, "abs_error": 0,
                  "children": 0, "children_exact": 0}  # fmt: skip
        weeks = [{"pages": [{"json_valid": True}] * pages}]
        return {"settings": {"engine": "ollama:m", "mode": "full"}, "counts": counts, "weeks": weeks, "latency_s": 10}  # fmt: skip

    combined = combine([run(1, 10, 1), run(9, 90, 3)])[("ollama:m", "full")]
    assert combined["cer"] == 0.1 and combined["pages"] == 4 and combined["scenarios"] == 2
