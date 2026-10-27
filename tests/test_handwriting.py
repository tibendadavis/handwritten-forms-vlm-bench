import datetime as dt

import pytest

from formbench.handwriting import Unreadable, completed_months, parse_age, parse_date


@pytest.mark.parametrize(
    "text, expected",
    [
        ("14-06-25", dt.date(2025, 6, 14)),
        ("14/06/25", dt.date(2025, 6, 14)),
        ("4-6-25", dt.date(2025, 6, 4)),
        ("04.06.2025", dt.date(2025, 6, 4)),
        (" 14 - 06 - 25 ", dt.date(2025, 6, 14)),
        ("14-06\n25", dt.date(2025, 6, 14)),
    ],
)
def test_parse_date_formats(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize("text", [None, "", "   "])
def test_blank_date_is_none(text):
    assert parse_date(text) is None


@pytest.mark.parametrize("text", ["32-01-25", "14-13-25", "29-02-25", "14-06", "June 14", "x"])
def test_unreadable_or_impossible_dates_raise(text):
    with pytest.raises(Unreadable):
        parse_date(text)


@pytest.mark.parametrize(
    "text, expected",
    [("7", (7, "years")), ("3/12", (3, "months")), ("6/52", (6, "weeks")), ("5/7", (5, "days"))],
)
def test_parse_age_notation(text, expected):
    assert parse_age(text) == expected


def test_age_blank_and_unreadable():
    assert parse_age("") is None
    with pytest.raises(Unreadable):
        parse_age("3/10")


@pytest.mark.parametrize(
    "birth, on, months",
    [
        (dt.date(2025, 1, 15), dt.date(2025, 2, 14), 0),
        (dt.date(2025, 1, 15), dt.date(2025, 2, 15), 1),
        (dt.date(2025, 1, 31), dt.date(2025, 7, 30), 5),
        (dt.date(2024, 6, 1), dt.date(2025, 6, 1), 12),
        (dt.date(2020, 6, 2), dt.date(2025, 6, 1), 59),
    ],
)
def test_completed_months(birth, on, months):
    assert completed_months(birth, on) == months
