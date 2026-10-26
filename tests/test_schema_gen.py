import pytest

from formbench.schema_gen import build_extraction_schema, build_fields, build_mapping, slugify

DEFAULT = {
    "id": "HllvX50cXC0",
    "name": "default",
    "categoryOptionCombos": [{"id": "HllvX50cXC0", "name": "default"}],
}
AGE = {
    "id": "ccAge",
    "name": "Age",
    "categoryOptionCombos": [{"id": "cocLt1", "name": "<1y"}, {"id": "cocGt1", "name": ">1y"}],
}


def de(uid, name, combo, value_type="INTEGER_ZERO_OR_POSITIVE", form_name=None):
    element = {"id": uid, "name": name, "valueType": value_type, "categoryCombo": combo}
    if form_name:
        element["formName"] = form_name
    return {"dataElement": element}


def dataset(elements, sections=None, attribute=DEFAULT):
    return {
        "id": "dsTest",
        "name": "Test form",
        "periodType": "Daily",
        "categoryCombo": attribute,
        "dataSetElements": elements,
        "sections": sections or [],
    }


def test_slugify_keeps_meaning_of_symbols():
    assert slugify("<1y") == "lt_1y"
    assert slugify(">1y") == "gt_1y"
    assert slugify("Malaria (RDT) tested") == "malaria_rdt_tested"


def test_category_option_combos_expand_to_unique_keys():
    fields = build_fields(dataset([de("deMal", "Malaria tested", AGE)]))
    assert [f.key for f in fields] == ["malaria_tested__lt_1y", "malaria_tested__gt_1y"]


def test_fields_follow_section_order():
    elements = [de("deA", "Alpha", DEFAULT), de("deB", "Beta", DEFAULT)]
    sections = [
        {"name": "Second", "sortOrder": 2, "dataElements": [{"id": "deA"}]},
        {"name": "First", "sortOrder": 1, "dataElements": [{"id": "deB"}]},
    ]
    fields = build_fields(dataset(elements, sections))
    assert [(f.key, f.section) for f in fields] == [("beta", "First"), ("alpha", "Second")]


def test_form_name_preferred_over_name():
    fields = build_fields(dataset([de("deA", "MAL - tested", DEFAULT, form_name="Malaria tested")]))
    assert fields[0].key == "malaria_tested"


def test_duplicate_keys_fail_loudly():
    with pytest.raises(ValueError, match="male"):
        build_fields(dataset([de("deA", "Male", DEFAULT), de("deB", "Male", DEFAULT)]))


def test_unsupported_value_type_fails_loudly():
    with pytest.raises(ValueError, match="FILE_RESOURCE"):
        build_fields(dataset([de("deA", "Photo", DEFAULT, value_type="FILE_RESOURCE")]))


def test_schema_requires_every_field_and_allows_null():
    ds = dataset([de("deMal", "Malaria tested", AGE)])
    schema = build_extraction_schema(ds, build_fields(ds))
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    assert schema["properties"]["malaria_tested__lt_1y"]["type"] == ["integer", "null"]
    assert "attribute_option" not in schema["properties"]


def test_attribute_options_become_enum():
    funding = {
        "id": "ccFund",
        "name": "Funding",
        "categoryOptionCombos": [
            {"id": "cocA", "name": "Partner A"},
            {"id": "cocB", "name": "Partner B"},
        ],
    }
    ds = dataset([de("deA", "Alpha", DEFAULT)], attribute=funding)
    schema = build_extraction_schema(ds, build_fields(ds))
    assert schema["properties"]["attribute_option"]["enum"] == ["Partner A", "Partner B", None]


def test_mapping_points_back_to_dhis2_uids():
    ds = dataset([de("deMal", "Malaria tested", AGE)])
    entry = build_mapping(ds, build_fields(ds))["fields"]["malaria_tested__lt_1y"]
    assert (entry["dataElement"], entry["categoryOptionCombo"]) == ("deMal", "cocLt1")
