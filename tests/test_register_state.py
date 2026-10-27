import datetime as dt

import pytest
from conftest import coc_uid
from test_register_aggregate import page, row

from formbench.register_spec import load_spec
from formbench.register_state import (
    RegisterState,
    book_id,
    corrections,
    discrepancies,
    page_identity,
)

GO_LIVE, TODAY = dt.date(2025, 6, 2), dt.date(2025, 6, 27)
OPV1_F = coc_uid("Dose 1, 0-5 months, Female")


@pytest.fixture
def spec(ri_metadata):
    return load_spec(ri_metadata)


def test_identity_uses_cohort_and_first_serial():
    assert page_identity(page(row("18"), row("19")), "x") == "2025-04-s18"
    assert page_identity(page(row("1"), month="??"), "IMG_1") == "unknown-IMG_1"
    assert book_id("MFL 123", "Outreach/Mobile") == "mfl_123_outreach_mobile"


def test_new_writing_in_empty_cell_is_fine_but_changes_are_flagged():
    approved = page(row("1", opv1="12-06-25"), row("2", opv1="12-06-25"))
    now = page(
        row("1", opv1="12/6/25", opv2="19-06-25"),  # same date, plus a new dose: fine
        row("2", opv1="17-06-25"),  # misread now or then
    )
    flags = discrepancies(now, approved)
    assert [(f.code, f.serial, f.column) for f in flags] == [
        ("changed_since_approved", "2", "opv1")
    ]
    assert flags[0].blocking


def test_row_that_disappeared_is_flagged():
    flags = discrepancies(page(row("1")), page(row("1"), row("2", opv1="12-06-25")))
    assert [(f.code, f.serial) for f in flags] == [("row_missing", "2")]


def test_late_dose_for_approved_week_becomes_full_day_correction(spec, tmp_path):
    state = RegisterState("book", root=tmp_path)
    week24 = [
        {**page(row("1", opv1="12-06-25"), row("2", opv1="12-06-25")), "page_id": "2025-04-s1"}
    ]
    state.approve("2025W24", week24, "Alice", "run-1", "2025-06-16")

    # Week 25 photo: a third child's OPV1 dated 12 June was written in late.
    now = [{**page(row("1", opv1="12-06-25"), row("2", opv1="12-06-25"), row("3", opv1="12-06-25")),
            "page_id": "2025-04-s1"}]  # fmt: skip
    totals, _ = corrections(now, state, spec, go_live=GO_LIVE, today=TODAY)
    assert totals[(dt.date(2025, 6, 12), "deOPV", OPV1_F)] == 3  # new full total, not +1


def test_correction_keeps_other_approved_pages_in_the_total(spec, tmp_path):
    state = RegisterState("book", root=tmp_path)
    april = {**page(row("1", opv1="12-06-25")), "page_id": "2025-04-s1"}
    march = {
        **page(row("1", birth="02-03-25", opv2="12-06-25"), month="March"),
        "page_id": "2025-03-s1",
    }
    state.approve("2025W24", [april, march], "Alice", "run-1", "2025-06-16")

    late = {**page(row("1", opv1="12-06-25"), row("2", opv1="12-06-25")), "page_id": "2025-04-s1"}
    totals, _ = corrections([late], state, spec, go_live=GO_LIVE, today=TODAY)
    day = dt.date(2025, 6, 12)
    assert totals[(day, "deOPV", OPV1_F)] == 2
    # The March page was not photographed again, but its dose still belongs to that day.
    assert totals[(day, "deOPV", coc_uid("Dose 2, 0-5 months, Female"))] == 1


def test_no_correction_when_nothing_new(spec, tmp_path):
    state = RegisterState("book", root=tmp_path)
    april = {**page(row("1", opv1="12-06-25")), "page_id": "2025-04-s1"}
    state.approve("2025W24", [april], "Alice", "run-1", "2025-06-16")
    with_new_week = {**page(row("1", opv1="12-06-25", opv2="20-06-25")), "page_id": "2025-04-s1"}
    assert corrections([with_new_week], state, spec, go_live=GO_LIVE, today=TODAY)[0] == {}
