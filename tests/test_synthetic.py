import datetime as dt
import random

import pytest
from test_synthetic_template import draw_register

from formbench.register.handwriting import parse_date
from formbench.synthetic.render import FONT_DIR, make_writer, photograph, render_page
from formbench.synthetic.simulate import (
    VISITS,
    age_notation,
    build_truth,
    fmt_date,
    private_fields,
    reading,
    simulate,
)
from formbench.synthetic.template import detect_grid

GO_LIVE = dt.date(2026, 9, 7)
COVER = {"facility_name": "X", "mfl_code": "MFL-1", "facility_type": "Outrech",
         "district": "", "region": ""}  # fmt: skip


@pytest.fixture(scope="module")
def sim():
    return simulate(3, 3, 3, GO_LIVE)


def test_same_seed_same_scenario():
    a, b = simulate(7, 3, 3, GO_LIVE), simulate(7, 3, 3, GO_LIVE)
    assert [[(c.birth, c.doses) for c in p.children] for p in a["book"]] == [
        [(c.birth, c.doses) for c in p.children] for p in b["book"]
    ]


def test_doses_follow_the_schedule(sim):
    order = [cols for _, cols in VISITS]
    for page in sim["book"][:-1]:  # the last page is the over-five catch-up child
        for child in page.children:
            days = [child.doses[c] for cols in order for c in cols if c in child.doses]
            assert days == sorted(days)
            assert all(d >= child.birth for d in days)
            for (_, a), (_, b) in zip(VISITS, VISITS[1:]):
                if a[0] in child.doses and b[0] in child.doses:
                    assert (child.doses[b[0]] - child.doses[a[0]]).days >= 28
            assert all(child.written[c] >= child.doses[c] for c in child.doses)


def test_edge_cases_are_injected(sim):
    codes = {e["code"] for e in sim["edges"]}
    assert {"birth_date_missing", "wrong_cohort", "future_date", "crossed_out"} <= codes


def test_truth_closes_every_week_with_reviewer_decisions(sim, ri_metadata):
    weeks = build_truth(sim, ri_metadata, COVER, ["dash"] * len(sim["book"]))
    assert [w["week"] for w in weeks] == sim["weeks"]
    assert all(w["decisions"]["attribute_option"] == "Outreach/Mobile" for w in weeks)
    assert sum(len(w["expected"]["values"]) for w in weeks) > 0
    for week in weeks:
        for photo in week["photos"]:
            for row in photo["reading"]["rows"]:
                assert "names" not in row and "phone" not in row


def test_reading_shows_only_what_was_written_by_then(sim):
    page = sim["book"][0]
    early = reading(page, GO_LIVE - dt.timedelta(days=400), "dash")
    late = reading(page, sim["end"], "dash")
    assert len(early["rows"]) <= len(late["rows"])
    for row in late["rows"]:
        for key, value in row.items():
            if key.startswith(("opv", "penta", "pcv", "ipv", "mcv", "rota", "bcg")) and value:
                parse_date(value)  # every written date is a real date


@pytest.mark.parametrize(
    "style, text", [("dash", "05-09-26"), ("slash", "05/09/26"), ("short", "5/9/26")]
)
def test_date_styles(style, text):
    assert fmt_date(dt.date(2026, 9, 5), style) == text
    assert parse_date(text) == dt.date(2026, 9, 5)


def test_age_notation():
    birth = dt.date(2026, 1, 1)
    assert age_notation(birth, dt.date(2026, 1, 4)) == "3/7"
    assert age_notation(birth, dt.date(2026, 1, 22)) == "3/52"
    assert age_notation(birth, dt.date(2026, 4, 1)) == "2/12"
    assert age_notation(birth, dt.date(2028, 3, 1)) == "2"


@pytest.mark.skipif(not any(FONT_DIR.glob("*.ttf")), reason="handwriting fonts not downloaded")
def test_render_and_photograph(sim):
    page, _, _ = draw_register()
    layout = detect_grid(page)
    layout["month_box"], layout["year_box"] = [600, 40, 1000, 140], [1100, 40, 1400, 140]
    fonts = [(p, 0.7) for p in sorted(FONT_DIR.glob("*.ttf"))]
    rng = random.Random(0)
    book_page = sim["book"][0]
    image = render_page(page, layout, reading(book_page, sim["end"], "dash"), private_fields(book_page),
                        make_writer(rng, fonts), rng, lambda d: fmt_date(d, "dash"))  # fmt: skip
    assert image.size == (page.shape[1], page.shape[0])
    for quality in ("scan", "good", "poor"):
        assert photograph(image, quality, rng).size[0] > 1000
