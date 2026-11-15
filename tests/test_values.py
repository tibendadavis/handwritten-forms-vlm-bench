import datetime as dt

import pytest

from formbench.values import (
    Unreadable,
    blank,
    clean_value,
    cohort,
    field_blank,
    parse_cohort,
    parse_date,
    same,
    same_value,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("12-06-26", dt.date(2026, 6, 12)),
        ("12/6/2026", dt.date(2026, 6, 12)),
        ("1.6.26", dt.date(2026, 6, 1)),
        ("12-06-\n26", dt.date(2026, 6, 12)),  # year wrapped onto a new line
        ("12 06 2026", dt.date(2026, 6, 12)),
        (" 3/4/25 ", dt.date(2025, 4, 3)),
    ],
)
def test_date_formats(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize("text", ["31-02-26", "12-13-26", "00-01-26"])
def test_impossible_dates_are_unreadable(text):
    with pytest.raises(Unreadable, match="impossible"):
        parse_date(text)


@pytest.mark.parametrize("text", ["June 12", "12-06", "12-06-126", "abc"])
def test_text_that_is_not_a_date_is_unreadable(text):
    with pytest.raises(Unreadable, match="not a date"):
        parse_date(text)


@pytest.mark.parametrize("text", [None, "", "   "])
def test_blank_date_is_none(text):
    assert parse_date(text) is None


@pytest.mark.parametrize(
    "month, year, expected",
    [
        ("April", "2025", (2025, 4)),
        ("apr.", "25", (2025, 4)),
        ("4", "2025", (2025, 4)),
        ("04", "25", (2025, 4)),
        ("SEPTEMBER", "2026", (2026, 9)),
    ],
)
def test_cohort_parsing(month, year, expected):
    assert parse_cohort(month, year) == expected


@pytest.mark.parametrize(
    "month, year", [("13", "2025"), ("0", "2025"), ("Aprl", "2025"), ("4", "202"), (None, "25")]
)
def test_unreadable_cohort(month, year):
    with pytest.raises(Unreadable):
        parse_cohort(month, year)
    assert cohort({"cohort_month": month, "cohort_year": year}) is None


def test_register_blank_and_clean():
    assert blank(None) and blank("  ") and not blank("0")
    assert clean_value(" NULL ") is None and clean_value("None") is None
    assert clean_value("") is None and clean_value(" 7 ") == "7" and clean_value(5) == "5"


def test_register_same_value():
    assert same_value(None, " ", "date")
    assert not same_value(None, "05-09-26", "date")
    assert same_value("05-09-26", "5/9/26", "date")
    assert not same_value("05-09-26", "06-09-26", "date")
    assert not same_value("05-09-26", "5 Sept", "date")  # unreadable side: not equal
    assert same_value("F", "female", "sex") and not same_value("F", "M", "sex")
    assert same_value("3 / 12", "3 /  12", "text") and same_value("Ab", "aB", "text")
    assert not same_value("3/12", "3/52", "text")


@pytest.mark.parametrize(
    "field_type, value",
    [
        ("count", None),
        ("count", ""),
        ("count", "null"),
        ("count", "None"),
        ("count", "0"),
        ("count", "0 0"),
        ("count", "000"),
        ("text", "null"),
    ],
)
def test_field_blank(field_type, value):
    assert field_blank(field_type, value)


@pytest.mark.parametrize("field_type, value", [("digits", "0"), ("text", "0"), ("count", "7")])
def test_field_not_blank(field_type, value):
    assert not field_blank(field_type, value)


def test_count_rules():
    assert same("count", "0", None)  # zero is the same as nothing written
    assert not same("count", "3", None)
    assert same("count", "047", "47") and same("count", "4 7", "47")
    assert not same("count", "47", "41")
    assert same("count", "1O", "1o")  # not digits: compared as text


def test_digits_keep_leading_zeros_but_ignore_separators():
    assert same("digits", "0012", "00-12")
    assert not same("digits", "0012", "12")


def test_mark_is_right_whenever_both_sides_are_set():
    assert same("mark", "signed", "stamp")
    assert not same("mark", "signed", None)


def test_date_falls_back_to_text():
    assert same("date", "01/06/26", "1-6-2026")
    assert same("date", "Week 3", "week  3")  # not dates: compared as text
    assert not same("date", "01/06/26", "Week 3")


def test_choice_and_text():
    assert same("choice", "Out-reach", "outreach")
    assert not same("choice", "North", "South")
    assert same("text", "EXAMPLE  HEALTH centre", "example health centre")
    assert not same("text", "EXAMPLEHEALTH", "example health")
