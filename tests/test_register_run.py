import datetime as dt

import pytest
from test_register_aggregate import page, row

from formbench.register_review import write_review
from formbench.register_run import compute_draft, names, resolve_org_unit
from formbench.register_spec import load_spec
from formbench.register_state import RegisterState

COVER = {"facility_name": "Amani HC", "mfl_code": "MFL-1", "facility_type": "Fixed/Facility"}
GO_LIVE, TODAY = dt.date(2025, 6, 2), dt.date(2025, 6, 27)


@pytest.fixture
def setup(ri_metadata, tmp_path):
    return load_spec(ri_metadata), names(ri_metadata), tmp_path


def draft(setup, pages, week="2025W24", cover=COVER, decisions=None):
    spec, labels, root = setup
    return compute_draft(
        pages, cover, spec, labels, week=week, go_live=GO_LIVE, today=TODAY,
        state_root=root / "state", decisions=decisions,
    )  # fmt: skip


def apr(*rows):
    return {**page(*rows), "page_id": "2025-04-s1"}


def test_draft_has_week_values_with_names_and_sources(setup):
    d = draft(setup, [apr(row("1", opv1="12-06-25"), row("2", opv1="20-06-25"))])
    assert d["attribute_option"] == {"name": "Fixed/Facility", "uid": "aoc0"}
    opv = [v for v in d["values"] if v["data_element_name"] == "OPV"]
    assert opv == [
        {"day": "2025-06-12", "data_element": "deOPV", "data_element_name": "OPV",
         "coc": opv[0]["coc"], "coc_name": "Dose 1, 0-5 months, Female", "value": 1,
         "sources": [["2025-04-s1", "1", "opv1"]]}
    ]  # fmt: skip
    assert d["unresolved_blocking"] == []


def test_blocking_flags_listed_until_resolved_by_decisions(setup):
    pages = [apr(row("1", birth=None, opv1="12-06-25"))]
    d = draft(setup, pages, cover={**COVER, "facility_type": "Outrech"})
    assert sorted(d["unresolved_blocking"]) == ["birth_date_missing:2025-04-s1:1", "facility_type_unclear"]  # fmt: skip
    fixed = draft(setup, pages, cover={**COVER, "facility_type": "Outrech"}, decisions={
        "attribute_option": "Outreach/Mobile",
        "pages": {"2025-04-s1": {"rows": {"1": {"birth_date": "10-04-25"}}}},
    })  # fmt: skip
    assert fixed["unresolved_blocking"] == [] and len(fixed["values"]) == 2


def test_doses_in_unapproved_earlier_week_are_noted_not_counted(setup):
    d = draft(setup, [apr(row("1", opv1="04-06-25", opv2="12-06-25"))])
    assert {v["day"] for v in d["values"]} == {"2025-06-12"}
    assert any(f["code"] == "earlier_week_unreported" for f in d["flags"])


def test_reread_differences_and_corrections(setup):
    spec, labels, root = setup
    state = RegisterState("mfl_1_fixed_facility", root=root / "state")
    state.approve("2025W24", [apr(row("1", opv1="12-06-25"))], "Alice", "r1", "2025-06-16")
    d = draft(setup, [apr(row("1", opv1="13-06-25"), row("2", opv1="12-06-25"))], week="2025W25")
    assert "changed_since_approved:2025-04-s1:1:opv1" in d["unresolved_blocking"]
    fix = {
        (c["day"], c["data_element_name"]): (c["previous"], c["value"]) for c in d["corrections"]
    }
    assert fix[("2025-06-12", "OPV")] == (1, 1)  # row 1 moved to 13 June, row 2 added on 12 June
    assert fix[("2025-06-13", "OPV")] == (None, 1)


def test_review_files(setup, tmp_path):
    d = draft(setup, [apr(row("1", birth=None, opv1="12-06-25"), row("2", opv1="12-06-25"))])
    readings = {"pages": [apr(row("1", birth=None, opv1="12-06-25"), row("2", opv1="12-06-25"))]}
    write_review(tmp_path, d, readings, ["Fixed/Facility", "Outreach/Mobile"])
    page_html = (tmp_path / "review.html").read_text()
    assert "must be resolved before approval" in page_html
    assert "id='row-2025-04-s1-2'" in page_html and 'href="#row-2025-04-s1-2"' in page_html
    decisions = (tmp_path / "decisions.yaml").read_text()
    assert "birth_date_missing:2025-04-s1:1" in decisions and "Outreach/Mobile" in decisions
    (tmp_path / "decisions.yaml").write_text("attribute_option: Fixed/Facility\n")
    write_review(tmp_path, d, readings)
    assert (tmp_path / "decisions.yaml").read_text() == "attribute_option: Fixed/Facility\n"


def test_blank_mfl_code_gives_null_org_unit_without_network():
    assert resolve_org_unit(None) == (None, "MFL Code is blank on the cover; orgUnit will be null")
