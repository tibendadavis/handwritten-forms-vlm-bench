from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

# DHIS2 valueType -> JSON Schema type the model must produce.
JSON_TYPES = {
    "INTEGER": "integer",
    "INTEGER_POSITIVE": "integer",
    "INTEGER_ZERO_OR_POSITIVE": "integer",
    "INTEGER_NEGATIVE": "integer",
    "NUMBER": "number",
    "PERCENTAGE": "number",
    "UNIT_INTERVAL": "number",
    "TEXT": "string",
    "LONG_TEXT": "string",
    "LETTER": "string",
    "BOOLEAN": "boolean",
    "TRUE_ONLY": "boolean",
    "DATE": "string",
}

# Keys reserved for the form header; no data field may use them.
HEADER_KEYS = ("period", "org_unit_name", "attribute_option")

# Symbols carry meaning in DHIS2 names ("<1y" vs ">1y") and must survive slugging.
SYMBOLS = [
    (">=", " gte "),
    ("<=", " lte "),
    ("≥", " gte "),
    ("≤", " lte "),
    ("<", " lt "),
    (">", " gt "),
    ("+", " plus "),
    ("&", " and "),
    ("%", " pct "),
]


def slugify(text: str) -> str:
    for symbol, word in SYMBOLS:
        text = text.replace(symbol, word)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    section: str | None
    data_element: str
    category_option_combo: str
    value_type: str


def _ordered_elements(ds: dict[str, Any]) -> list[tuple[dict[str, Any], str | None]]:
    """Data set elements in printed-form order: by section, then any left over."""
    by_id = {dse["dataElement"]["id"]: dse for dse in ds.get("dataSetElements", [])}
    ordered: list[tuple[dict[str, Any], str | None]] = []
    seen: set[str] = set()
    for section in sorted(ds.get("sections", []), key=lambda s: s.get("sortOrder", 0)):
        for ref in section.get("dataElements", []):
            if ref["id"] in by_id and ref["id"] not in seen:
                ordered.append((by_id[ref["id"]], section["name"]))
                seen.add(ref["id"])
    rest = [dse for uid, dse in by_id.items() if uid not in seen]
    rest.sort(key=lambda dse: dse["dataElement"]["name"])
    ordered.extend((dse, None) for dse in rest)
    return ordered


def _check_unique(fields: list[Field]) -> None:
    seen: set[str] = set()
    clashes = {f.key for f in fields if f.key in HEADER_KEYS or f.key in seen or seen.add(f.key)}
    if clashes:
        raise ValueError(f"Duplicate field keys: {', '.join(sorted(clashes))}")


def build_fields(ds: dict[str, Any]) -> list[Field]:
    fields: list[Field] = []
    for dse, section in _ordered_elements(ds):
        de = dse["dataElement"]
        if de["valueType"] not in JSON_TYPES:
            raise ValueError(
                f"Unsupported valueType {de['valueType']} on '{de['name']}' ({de['id']})"
            )
        combo = dse.get("categoryCombo") or de["categoryCombo"]
        base = de.get("formName") or de["name"]
        for coc in sorted(combo["categoryOptionCombos"], key=lambda c: c["name"]):
            is_default = coc["name"] == "default"
            fields.append(
                Field(
                    key=slugify(base) if is_default else f"{slugify(base)}__{slugify(coc['name'])}",
                    label=base if is_default else f"{base} ({coc['name']})",
                    section=section,
                    data_element=de["id"],
                    category_option_combo=coc["id"],
                    value_type=de["valueType"],
                )
            )
    _check_unique(fields)
    return fields


def _attribute_options(ds: dict[str, Any]) -> dict[str, str]:
    combo = ds.get("categoryCombo") or {}
    if combo.get("name", "default") == "default":
        return {}
    return {c["name"]: c["id"] for c in combo["categoryOptionCombos"]}


def build_extraction_schema(ds: dict[str, Any], fields: list[Field]) -> dict[str, Any]:
    props: dict[str, Any] = {
        "period": {
            "type": ["string", "null"],
            "description": "Date or period written on the form, exactly as written",
        },
        "org_unit_name": {
            "type": ["string", "null"],
            "description": "Facility name exactly as handwritten",
        },
    }
    options = _attribute_options(ds)
    if options:
        props["attribute_option"] = {
            "type": ["string", "null"],
            "enum": [*sorted(options), None],
            "description": f"{ds['categoryCombo']['name']}: the option marked on the form",
        }
    for f in fields:
        prop: dict[str, Any] = {"type": [JSON_TYPES[f.value_type], "null"], "description": f.label}
        if f.value_type == "DATE":
            prop["format"] = "date"
        props[f.key] = prop
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": ds["name"],
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def build_mapping(ds: dict[str, Any], fields: list[Field]) -> dict[str, Any]:
    return {
        "dataSet": {"id": ds["id"], "name": ds["name"], "periodType": ds["periodType"]},
        "attributeOptions": _attribute_options(ds),
        "fields": {
            f.key: {
                "dataElement": f.data_element,
                "categoryOptionCombo": f.category_option_combo,
                "valueType": f.value_type,
                "label": f.label,
                "section": f.section,
            }
            for f in fields
        },
    }
