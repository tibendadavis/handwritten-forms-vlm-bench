from jsonschema import Draft202012Validator

from formbench.register_schema import (
    cover_schema,
    page_prompt,
    page_schema,
    rows_prompt,
    rows_schema,
)
from formbench.register_spec import DOSE_COLUMNS

ROW = {
    "serial": "1",
    "card_number": "C-104",
    "sex": "F",
    "birth_date": "03-04-25",
    "age": "6/52",
    **{c.key: None for c in DOSE_COLUMNS},
}


def test_page_reply_validates():
    reply = {"cohort_month": "April", "cohort_year": "2025", "rows": [ROW]}
    assert not list(Draft202012Validator(page_schema()).iter_errors(reply))


def test_personal_data_cannot_be_returned():
    reply = {"rows": [{**ROW, "child_name": "Amina Ali Hassan"}]}
    errors = list(Draft202012Validator(rows_schema()).iter_errors(reply))
    assert errors and "child_name" in errors[0].message
    for schema in (page_schema(), cover_schema()):
        text = str(schema).lower()
        assert "phone" not in text and "guardian" not in text and "village" not in text


def test_every_dose_column_is_requested():
    row = rows_schema()["properties"]["rows"]["items"]
    assert {c.key for c in DOSE_COLUMNS} <= set(row["required"])
    for prompt in (page_prompt(), rows_prompt()):
        assert all(f"- {c.key}: column {c.letter} " in prompt for c in DOSE_COLUMNS)
        assert "Do not transcribe child names" in prompt
