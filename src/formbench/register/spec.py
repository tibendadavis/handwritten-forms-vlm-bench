from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DoseColumn:
    letter: str
    key: str
    label: str
    vaccine: str | None  # DHIS2 data element name; None = read but not counted
    dose: int | None


# register dose columns. Vaccine names match the data element names; UIDs are
# resolved from the data set metadata at runtime.
DOSE_COLUMNS = (
    DoseColumn("I", "bcg", "BCG", "BCG", 0),
    DoseColumn("J", "polio", "Polio", None, None),  # mapping unresolved, see open questions
    DoseColumn("K", "opv0", "OPV 0 (birth)", "OPV", 0),
    DoseColumn("L", "opv1", "OPV 1", "OPV", 1),
    DoseColumn("M", "opv2", "OPV 2", "OPV", 2),
    DoseColumn("N", "opv3", "OPV 3", "OPV", 3),
    DoseColumn("O", "penta1", "Pentavalent 1", "Pentavalent", 1),
    DoseColumn("P", "penta2", "Pentavalent 2", "Pentavalent", 2),
    DoseColumn("Q", "penta3", "Pentavalent 3", "Pentavalent", 3),
    DoseColumn("R", "ipv1", "IPV 1", "IPV", 1),
    DoseColumn("S", "ipv2", "IPV 2", "IPV", 2),
    DoseColumn("T", "pcv1", "PCV 1", "PCV", 1),
    DoseColumn("U", "pcv2", "PCV 2", "PCV", 2),
    DoseColumn("V", "pcv3", "PCV 3", "PCV", 3),
    DoseColumn("W", "rota1", "Rotavirus 1", "Rota", 1),
    DoseColumn("X", "mcv1", "MCV 1", "MRI", 1),
    DoseColumn("Y", "mcv2", "MCV 2", "MRI", 2),
)
CHILDREN_VACCINATED = "Children vaccinated"

_DOSE = re.compile(r"^Dose (\d+)$")
_AGE = re.compile(r"(\d+)-(\d+) months")
SEXES = {"Female": "F", "Male": "M"}


@dataclass(frozen=True)
class Disaggregation:
    """What a category option combo name says: dose, sex and age band (inclusive months)."""

    dose: int | None = None
    sex: str | None = None
    months: tuple[int, int] | None = None

    def matches(self, dose: int | None, sex: str, months: int) -> bool:
        return (
            self.dose == dose
            and self.sex == sex
            and self.months is not None
            and self.months[0] <= months <= self.months[1]
        )


def parse_coc(name: str) -> Disaggregation:
    """'Dose 1, 6-11 months, Female' -> Disaggregation(1, 'F', (6, 11)). Unknown parts raise."""
    dose = sex = months = None
    for part in name.split(", "):
        if _DOSE.match(part):
            dose = int(_DOSE.match(part).group(1))
        elif part in SEXES:
            sex = SEXES[part]
        elif _AGE.search(part):
            low, high = _AGE.search(part).groups()
            months = (int(low), int(high))
        else:
            raise ValueError(f"Cannot interpret category option combo part {part!r} in {name!r}")
    return Disaggregation(dose, sex, months)


@dataclass(frozen=True)
class Target:
    """A data element and its category option combos, as (disaggregation, COC UID)."""

    data_element: str
    combos: tuple[tuple[Disaggregation, str], ...]

    def coc(self, dose: int | None, sex: str, months: int) -> str | None:
        """The COC for this dose, sex and age; None when the metadata has no band for it."""
        hits = [uid for d, uid in self.combos if d.matches(dose, sex, months)]
        if len(hits) > 1:
            raise ValueError(f"Overlapping age bands in {self.data_element}")
        return hits[0] if hits else None


@dataclass(frozen=True)
class RegisterSpec:
    dataset_id: str
    period_type: str
    vaccines: dict[str, Target]
    children: Target
    attribute_options: dict[str, str]


def _element_name(de: dict[str, Any]) -> str:
    return de.get("formName") or de["name"]


def load_spec(ds: dict[str, Any]) -> RegisterSpec:
    """Resolve the register's vaccines to data elements and parse every COC; fail loudly."""
    by_name: dict[str, list[dict[str, Any]]] = {}
    for dse in ds["dataSetElements"]:
        by_name.setdefault(_element_name(dse["dataElement"]).casefold(), []).append(dse)

    def target(name: str) -> Target:
        found = by_name.get(name.casefold(), [])
        if len(found) != 1:
            raise ValueError(f"Expected one data element named {name!r}, found {len(found)}")
        dse = found[0]
        combo = dse.get("categoryCombo") or dse["dataElement"]["categoryCombo"]
        cocs = sorted(combo["categoryOptionCombos"], key=lambda c: c["name"])
        return Target(
            dse["dataElement"]["id"], tuple((parse_coc(c["name"]), c["id"]) for c in cocs)
        )

    vaccines = {c.vaccine: target(c.vaccine) for c in DOSE_COLUMNS if c.vaccine}
    combo = ds.get("categoryCombo") or {}
    options = {} if combo.get("name", "default") == "default" else combo["categoryOptionCombos"]
    return RegisterSpec(
        dataset_id=ds["id"],
        period_type=ds["periodType"],
        vaccines=vaccines,
        children=target(CHILDREN_VACCINATED),
        attribute_options={o["name"]: o["id"] for o in options},
    )
