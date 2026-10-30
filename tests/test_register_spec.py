import pytest
from conftest import coc_uid

from formbench.register.spec import DOSE_COLUMNS, Disaggregation, load_spec, parse_coc


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Dose 1, 6-11 months, Female", Disaggregation(1, "F", (6, 11))),
        ("Dose 0, 0-5 months, Male", Disaggregation(0, "M", (0, 5))),
        (
            "Female, 0-11 months default or 6-11 months (zero-dose)",
            Disaggregation(None, "F", (0, 11)),
        ),
        ("Male, 24-59 months", Disaggregation(None, "M", (24, 59))),
    ],
)
def test_parse_coc(name, expected):
    assert parse_coc(name) == expected


def test_unknown_coc_part_fails_loudly():
    with pytest.raises(ValueError, match="Issued"):
        parse_coc("Issued")


def test_every_counted_column_resolves_to_a_data_element(ri_metadata):
    spec = load_spec(ri_metadata)
    assert set(spec.vaccines) == {c.vaccine for c in DOSE_COLUMNS if c.vaccine}
    assert spec.vaccines["OPV"].data_element == "deOPV"
    assert spec.attribute_options["Outreach/Mobile"] == "aoc1"


def test_coc_lookup_uses_age_band_on_the_dose_date(ri_metadata):
    opv = load_spec(ri_metadata).vaccines["OPV"]
    assert opv.coc(1, "F", 2) == coc_uid("Dose 1, 0-5 months, Female")
    assert opv.coc(1, "F", 6) == coc_uid("Dose 1, 6-11 months, Female")
    assert opv.coc(3, "M", 59) == coc_uid("Dose 3, 24-59 months, Male")


def test_no_band_means_no_coc(ri_metadata):
    spec = load_spec(ri_metadata)
    assert spec.vaccines["MRI"].coc(1, "F", 60) is None
    assert spec.children.coc(None, "M", 11) == coc_uid(
        "Male, 0-11 months default or 6-11 months (zero-dose)"
    )


def test_missing_data_element_fails_loudly(ri_metadata):
    ri_metadata["dataSetElements"] = [
        dse for dse in ri_metadata["dataSetElements"] if dse["dataElement"]["name"] != "PCV"
    ]
    with pytest.raises(ValueError, match="PCV"):
        load_spec(ri_metadata)
