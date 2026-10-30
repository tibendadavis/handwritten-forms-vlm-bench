import datetime as dt

import pytest
from conftest import coc_uid

from formbench.register.aggregate import (
    aggregate,
    apply_decisions,
    classify_days,
    iso_week_days,
    parse_cohort,
    resolve_attribute_option,
    select_days,
)
from formbench.register.spec import DOSE_COLUMNS, load_spec

GO_LIVE = dt.date(2025, 6, 2)
TODAY = dt.date(2025, 6, 20)
D = dt.date


def row(serial, sex="F", birth="10-04-25", **doses):
    blank = {c.key: None for c in DOSE_COLUMNS}
    return {"serial": serial, "card_number": None, "sex": sex, "birth_date": birth, "age": None,
            **blank, **doses}  # fmt: skip


def page(*rows, month="April", year="2025", pid="p-apr"):
    return {"page_id": pid, "cohort_month": month, "cohort_year": year, "rows": list(rows)}


@pytest.fixture
def spec(ri_metadata):
    return load_spec(ri_metadata)


def run(spec, *pages, resolved=frozenset()):
    return aggregate(list(pages), spec, go_live=GO_LIVE, today=TODAY, resolved=resolved)


def codes(agg):
    return sorted(f.code for f in agg.flags)


def test_counts_doses_on_their_own_date_with_age_band(spec):
    # Born 10 April: on 12 June the child is 2 completed months old -> 0-5 months band.
    agg = run(
        spec, page(row("1", opv1="12-06-25", penta1="12-06-25"), row("2", sex="M", opv1="12-06-25"))
    )
    assert agg.totals[(D(2025, 6, 12), "deOPV", coc_uid("Dose 1, 0-5 months, Female"))] == 1
    assert agg.totals[(D(2025, 6, 12), "deOPV", coc_uid("Dose 1, 0-5 months, Male"))] == 1
    assert agg.totals[(D(2025, 6, 12), "dePentavalent", coc_uid("Dose 1, 0-5 months, Female"))] == 1
    assert agg.flags == []


def test_bcg_and_opv0_are_dose_zero(spec):
    agg = run(spec, page(row("1", birth="03-06-25", bcg="03-06-25", opv0="04-06-25"), month="6"))
    assert agg.totals[(D(2025, 6, 3), "deBCG", coc_uid("Dose 0, 0-5 months, Female"))] == 1
    assert agg.totals[(D(2025, 6, 4), "deOPV", coc_uid("Dose 0, 0-5 months, Female"))] == 1


def test_children_vaccinated_counts_children_not_doses(spec):
    one = row("1", opv1="12-06-25", penta1="12-06-25", pcv1="12-06-25", rota1="12-06-25")
    two = row("2", opv1="12-06-25")
    agg = run(spec, page(one, two))
    key = (
        D(2025, 6, 12),
        "deChildren",
        coc_uid("Female, 0-11 months default or 6-11 months (zero-dose)"),
    )
    assert agg.totals[key] == 2
    assert sorted(agg.provenance[key]) == [("p-apr", "1", "*"), ("p-apr", "2", "*")]


def test_provenance_points_to_cells(spec):
    agg = run(spec, page(row("7", opv1="12-06-25")))
    key = (D(2025, 6, 12), "deOPV", coc_uid("Dose 1, 0-5 months, Female"))
    assert agg.provenance[key] == [("p-apr", "7", "opv1")]


def test_polio_column_is_listed_not_counted(spec):
    agg = run(spec, page(row("1", polio="12-06-25")))
    assert agg.totals == {} and codes(agg) == ["unmapped_column"]
    assert not agg.flags[0].blocking


def test_before_go_live_listed_not_counted(spec):
    agg = run(spec, page(row("1", opv1="22-05-25", opv2="12-06-25")))
    assert [k[0] for k in agg.totals] == [D(2025, 6, 12), D(2025, 6, 12)]
    assert codes(agg) == ["before_go_live"]


def test_missing_birth_date_blocks_the_row(spec):
    agg = run(spec, page(row("1", birth=None, opv1="12-06-25")))
    assert agg.totals == {} and codes(agg) == ["birth_date_missing"]
    assert agg.flags[0].blocking


def test_reviewer_birth_date_unblocks_the_row(spec):
    pages = apply_decisions(
        [page(row("1", birth=None, opv1="12-06-25"))],
        {"pages": {"p-apr": {"rows": {"1": {"birth_date": "10-04-25"}}}}},
    )
    assert run(spec, *pages).flags == []


def test_wrong_cohort_not_counted_until_accepted(spec):
    stray = page(row("1", birth="10-03-25", opv1="12-06-25"))
    assert codes(run(spec, stray)) == ["wrong_cohort"]
    assert run(spec, stray).totals == {}
    accepted = apply_decisions(
        [stray], {"pages": {"p-apr": {"rows": {"1": {"accept_cohort": True}}}}}
    )
    assert len(run(spec, *accepted).totals) == 2  # the dose and the child


def test_age_without_band_is_not_counted(spec):
    old = page(row("1", birth="01-05-20", mcv2="12-06-25"), month="5", year="20")
    agg = run(spec, old)
    assert agg.totals == {} and codes(agg) == ["no_age_band"]
    assert not agg.flags[0].blocking


@pytest.mark.parametrize(
    "doses, code",
    [
        ({"opv1": "31-06-25"}, "date_unreadable"),
        ({"opv1": "30-06-25"}, "future_date"),
        ({"opv1": "03-04-25"}, "before_go_live"),
    ],
)
def test_bad_dates_are_flagged(spec, doses, code):
    assert code in codes(run(spec, page(row("1", **doses))))


def test_dose_before_birth_blocks(spec):
    agg = run(spec, page(row("1", birth="15-04-25", opv1="05-06-25", opv2="10-06-25")))
    assert agg.flags == []
    agg = run(spec, page(row("1", birth="15-06-25", opv1="05-06-25"), month="6"))
    assert codes(agg) == ["before_birth"]


def test_sex_and_cohort_problems_block(spec):
    assert codes(run(spec, page(row("1", sex=None, opv1="12-06-25")))) == ["sex_unreadable"]
    assert codes(run(spec, page(row("1", opv1="12-06-25"), month="Smarch"))) == [
        "cohort_unreadable"
    ]


def test_duplicate_serial_blocks_until_resolved(spec):
    dup = page(row("4", opv1="12-06-25"), row("4", opv1="12-06-25"))
    agg = run(spec, dup)
    assert codes(agg) == ["duplicate_serial"]
    key = (D(2025, 6, 12), "deOPV", coc_uid("Dose 1, 0-5 months, Female"))
    assert agg.totals[key] == 1
    assert run(spec, dup, resolved=frozenset({agg.flags[0].id})).totals[key] == 2


def test_blank_cells_produce_no_values(spec):
    agg = run(spec, page(row("1"), row("2", opv1="")))
    assert agg.totals == {} and agg.flags == []


def test_week_selection_and_classification(spec):
    agg = run(spec, page(row("1", opv1="03-06-25", opv2="12-06-25", opv3="17-06-25")))
    week = iso_week_days("2025W24")
    assert week[0] == D(2025, 6, 9) and week[-1] == D(2025, 6, 15)
    assert {k[0] for k in select_days(agg, week)} == {D(2025, 6, 12)}
    days = classify_days(agg, "2025W24", approved_weeks={"2025W23"})
    assert days == {"week": {D(2025, 6, 12)}, "approved": {D(2025, 6, 3)}, "unreported": set(),
                    "later": {D(2025, 6, 17)}}  # fmt: skip
    assert classify_days(agg, "2025W24", set())["unreported"] == {D(2025, 6, 3)}


@pytest.mark.parametrize(
    "month, year, expected",
    [("April", "2025", (2025, 4)), ("apr.", "25", (2025, 4)), ("04", "2025", (2025, 4))],
)
def test_parse_cohort(month, year, expected):
    assert parse_cohort(month, year) == expected


@pytest.mark.parametrize(
    "written, expected",
    [
        ("Fixed/Facility", "aoc0"),
        ("fixed facility", "aoc0"),
        ("Outreach", "aoc1"),
        ("HEALTH CAMP", "aoc2"),
    ],
)
def test_facility_type_matches_attribute_option(spec, written, expected):
    assert resolve_attribute_option(written, spec) == (expected, None)


@pytest.mark.parametrize("written", [None, "", "Outrech", "Unclear", "Not filled"])
def test_unclear_facility_type_needs_a_reviewer(spec, written):
    uid, flag = resolve_attribute_option(written, spec)
    assert uid is None and flag.blocking and flag.code == "facility_type_unclear"
    assert resolve_attribute_option(written, spec, chosen="Not filled") == ("aoc4", None)


def test_misspelling_gets_a_suggestion(spec):
    _, flag = resolve_attribute_option("Outrech/Mobil", spec)
    assert "closest option: Outreach/Mobile" in flag.message
