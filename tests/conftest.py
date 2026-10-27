import pytest

AGE_BANDS = ["0-5 months", "6-11 months", "12-23 months", "24-59 months"]
CHILD_BANDS = ["0-11 months default or 6-11 months (zero-dose)", "12-23 months", "24-59 months"]
VACCINES = ["BCG", "OPV", "Pentavalent", "IPV", "PCV", "Rota", "MRI"]
SESSIONS = ["Fixed/Facility", "Outreach/Mobile", "Health camp", "Unclear", "Not filled"]


def coc_uid(name):
    return "coc_" + "".join(ch for ch in name if ch.isalnum())


def _combo(uid, names):
    return {
        "id": uid,
        "name": uid,
        "categoryOptionCombos": [{"id": coc_uid(n), "name": n} for n in names],
    }


def _element(uid, name, combo):
    return {"dataElement": {"id": uid, "name": name, "valueType": "NUMBER", "categoryCombo": combo}}


@pytest.fixture
def ri_metadata():
    """A small stand-in for the RI v2 data set with the same naming conventions."""
    doses = _combo(
        "ccDose",
        [
            f"Dose {d}, {band}, {sex}"
            for d in range(4)
            for band in AGE_BANDS
            for sex in ("Female", "Male")
        ],
    )
    children = _combo(
        "ccChild", [f"{sex}, {band}" for sex in ("Female", "Male") for band in CHILD_BANDS]
    )
    stock = _combo("ccStock", ["Issued", "Returned/Remaining"])
    elements = [_element(f"de{v}", v, doses) for v in VACCINES]
    elements += [
        _element("deChildren", "Children vaccinated", children),
        _element("deStock", "OPV Inventory", stock),
    ]
    return {
        "id": "dsRI",
        "name": "RI test",
        "periodType": "Daily",
        "categoryCombo": {
            "id": "ccSession",
            "name": "Session type",
            "categoryOptionCombos": [{"id": f"aoc{i}", "name": n} for i, n in enumerate(SESSIONS)],
        },
        "dataSetElements": elements,
        "sections": [],
    }
